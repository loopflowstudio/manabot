//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::ability::{
    Ability, Actor, Aggregate, AttackSize, CompareOp, Condition, DamageKind, DamageRecipientFilter,
    Effect, EffectValue, KeywordAction, ObjectProperty, ObjectSelector, PaymentCost, Per,
    PlayWindow, PlayerSelector, Scope, TargetRequirement, TargetSpec, TriggerCondition,
    TriggerSubject, TurnStep,
};
use managym_state::card::{
    ActivatedAbilityDefinition, ActivationZone, AdditionalCost, CardDefinition, CardType,
    CardTypes, DamageTransform, Keyword, Keywords, PrintedCharacteristics, ReplacementEffect,
    Restriction, SacrificeCost, SpellCostReduction, StaticAbility, StaticRestriction, StaticScope,
    TargetCostReduction,
};
use managym_state::effect::Modification;
use managym_state::mana::{Color, ManaCost};
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;
use managym_state::zone::ZoneType;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        fishing_pole(),
        ravenous_amulet(),
        hare_apparent(),
        ingenious_leonin(),
        tragic_banshee(),
        vengeful_bloodwitch(),
        needletooth_pack(),
        slumbering_cerberus(),
        wardens_of_the_cycle(),
        courageous_goblin(),
        bulk_up(),
        ashroot_animist(),
        alesha_who_laughs_at_fate(),
        searslicer_goblin(),
        gorehorn_raider(),
        goblin_boarders(),
        battlesong_berserker(),
        midnight_snack(),
        skyship_buccaneer(),
        armasaur_guide(),
        blasphemous_edict(),
        cackling_prowler(),
        cat_collector(),
        chart_a_course(),
        claws_out(),
        drake_hatcher(),
        erudite_wizard(),
        felling_blow(),
        fiendish_panda(),
        high_society_hunter(),
        homunculus_horde(),
        hungry_ghoul(),
        incinerating_blast(),
        herald_of_eternal_dawn(),
        kellan_planar_trailblazer(),
        linden_the_steadfast_queen(),
        luminous_rebuke(),
        lunar_insight(),
        mischievous_mystic(),
        leyline_axe(),
        sire_of_seven_deaths(),
        vanguard_seraph(),
        bloodthirsty_conqueror(),
        exemplar_of_light(),
        scrawling_crawler(),
        firespitter_whelp(),
        archmage_of_runes(),
        goblin_negotiation(),
        niv_mizzet_visionary(),
        kiora_the_rising_tide(),
        crypt_feaster(),
        revenge_of_the_rats(),
        spinner_of_souls(),
        dawnwing_marshal(),
        treetop_snarespinner(),
        apothecary_stomper(),
        sower_of_chaos(),
        ambush_wolf(),
        beast_kin_ranger(),
        bigfin_bouncer(),
        crackling_cyclops(),
        dauntless_veteran(),
        dazzling_angel(),
        elementalist_adept(),
        felidar_savior(),
        hinterland_sanctifier(),
        sanguine_syphoner(),
        boltwave(),
        faebloom_trick(),
        joust_through(),
        preposterous_proportions(),
        refute(),
        uncharted_voyage(),
        strix_lookout(),
        icewind_elemental(),
        rune_sealed_wall(),
        starlight_snare(),
        skyknight_squire(),
        inspiring_paladin(),
        billowing_shriekmass(),
        quick_draw_katana(),
        twinblade_blessing(),
        elvish_regrower(),
        clinquant_skymage(),
        seekers_folly(),
        goblin_surprise(),
        stab(),
        quakestrider_ceratops(),
        eager_trufflesnout(),
        infestation_sage(),
        elfsworn_giant(),
        rite_of_the_dragoncaller(),
        dragon_trainer(),
        guarded_heir(),
        anthem_of_champions(),
        arahbo_the_first_fang(),
        helpful_hunter(),
        lightshell_duo(),
        prideful_parent(),
        high_fae_trickster(),
        sphinx_of_forgotten_lore(),
        suspicious_shambler(),
        tinybones_bauble_burglar(),
        loot_exuberant_explorer(),
        quilled_greatwurm(),
        redcap_gutter_dweller(),
        zul_ashur_lich_lord(),
        twinflame_tyrant(),
        fleeting_flight(),
        grappling_kraken(),
    ]
}

/// Threshold (ability word, CR 207.2c): "if there are seven or more cards in your
/// graveyard", checked as the trigger would trigger and again as it
/// resolves (CR 603.4).
fn threshold() -> Condition {
    Condition::GraveyardAtLeast {
        count: 7,
        predicate: CardPredicate::default(),
    }
}

fn kiora_the_rising_tide() -> CardDefinition {
    let text_box = "When Kiora enters, draw two cards, then discard two cards.\nThreshold — Whenever Kiora attacks, if there are seven or more cards in your graveyard, you may create Scion of the Deep, a legendary 8/8 blue Octopus creature token.";
    CardDefinition {
        key: "fdn.kiora_the_rising_tide",
        scryfall_id: "83f20a32-9f5d-4a68-8995-549e57554da2",
        printed: PrintedCharacteristics {
            name: "Kiora, the Rising Tide".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Merfolk".into(), "Noble".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![
                        Effect::DrawCards {
                            count: EffectValue::Fixed(2),
                        },
                        Effect::Discard {
                            what: ObjectSelector::You,
                            count: 2,
                        },
                    ],
                },
                Ability::Triggered {
                    condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                    intervening_if: Some(threshold()),
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::default()),
                        if_paid: vec![Effect::create_tokens("token.scion_of_the_deep", 1)],
                        if_declined: vec![],
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn crypt_feaster() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nThreshold — Whenever this creature attacks, if there are seven or more cards in your graveyard, this creature gets +2/+0 until end of turn.";
    CardDefinition {
        key: "fdn.crypt_feaster",
        scryfall_id: "3b072811-998a-4a71-b59c-6afecc0dc4b6",
        printed: PrintedCharacteristics {
            name: "Crypt Feaster".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Menace]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: Some(threshold()),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(2, 0),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn revenge_of_the_rats() -> CardDefinition {
    let text_box = "Create a tapped 1/1 black Rat creature token for each creature card in your graveyard.\nFlashback {2}{B}{B} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "fdn.revenge_of_the_rats",
        scryfall_id: "1f463c55-39a0-4f2f-aae3-0c5540bde5b7",
        printed: PrintedCharacteristics {
            name: "Revenge of the Rats".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{2}{B}{B}"))],
            // Counted as it resolves; cast with flashback, the spell is on
            // the stack and not among the creature cards (CR 702.34a).
            spell_effects: vec![Effect::CreateToken {
                token: "token.black_rat".into(),
                count: EffectValue::GraveyardMatching {
                    base: 0,
                    predicate: CardPredicate::creature(),
                },
                tapped_and_attacking: false,
                tapped: true,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vanguard_seraph() -> CardDefinition {
    let text_box = "Flying\nWhenever you gain life for the first time each turn, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "fdn.vanguard_seraph",
        scryfall_id: "4329c861-fc16-4a96-9c03-25af6ac2adc8",
        printed: PrintedCharacteristics {
            name: "Vanguard Seraph".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::GainsLife {
                    who: Actor::You,
                    nth: Some(1),
                },
                intervening_if: None,
                effects: vec![Effect::Surveil { count: 1 }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bloodthirsty_conqueror() -> CardDefinition {
    let text_box = "Flying, deathtouch\nWhenever an opponent loses life, you gain that much life. (Damage causes loss of life.)";
    CardDefinition {
        key: "fdn.bloodthirsty_conqueror",
        scryfall_id: "ce860ed4-a5bd-4347-9eab-dd716ea84db1",
        printed: PrintedCharacteristics {
            name: "Bloodthirsty Conqueror".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Knight".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Deathtouch]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::LosesLife {
                    who: Actor::Opponent,
                },
                intervening_if: None,
                effects: vec![Effect::gain_life(EffectValue::TriggeringAmount)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn exemplar_of_light() -> CardDefinition {
    let text_box = "Flying\nWhenever you gain life, put a +1/+1 counter on this creature.\nWhenever you put one or more +1/+1 counters on this creature, draw a card. This ability triggers only once each turn.";
    CardDefinition {
        key: "fdn.exemplar_of_light",
        scryfall_id: "920c8fc5-fdd2-446a-a676-5c363f96928f",
        printed: PrintedCharacteristics {
            name: "Exemplar of Light".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::GainsLife {
                        who: Actor::You,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::OnceEachTurn {
                        condition: Box::new(TriggerCondition::PutsCountersOn {
                            who: Actor::You,
                            kind: CounterKind::PlusOnePlusOne,
                            subject: TriggerSubject::This,
                        }),
                    },
                    intervening_if: None,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sire_of_seven_deaths() -> CardDefinition {
    let text_box = "Reach, first strike\nVigilance, menace\nTrample, lifelink\nWard—Pay 7 life.";
    CardDefinition {
        key: "fdn.sire_of_seven_deaths",
        scryfall_id: "8d8432a7-1c8a-4cfb-947c-ecf9791063eb",
        printed: PrintedCharacteristics {
            name: "Sire of Seven Deaths".into(),
            mana_cost: Some(ManaCost::parse("{7}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Eldrazi".into()],
            power: Some(7),
            toughness: Some(7),
            keywords: Keywords::of(&[
                Keyword::Reach,
                Keyword::FirstStrike,
                Keyword::Vigilance,
                Keyword::Menace,
                Keyword::Trample,
                Keyword::Lifelink,
            ]),
            statics: vec![StaticAbility::Ward(PaymentCost::Life(7))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hungry_ghoul() -> CardDefinition {
    let text_box = "{1}, Sacrifice another creature: Put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "fdn.hungry_ghoul",
        scryfall_id: "790f9433-7565-4f7f-88e8-8af762ea0296",
        printed: PrintedCharacteristics {
            name: "Hungry Ghoul".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into()],
            power: Some(2),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                    another: true,
                    or_mana: None,
                }),
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kellan_planar_trailblazer() -> CardDefinition {
    let text_box = "{1}{R}: If Kellan is a Scout, it becomes a Human Faerie Detective and gains \"Whenever Kellan deals combat damage to a player, exile the top card of your library. You may play that card this turn.\"\n{2}{R}: If Kellan is a Detective, it becomes a 3/2 Human Faerie Rogue and gains double strike.";
    CardDefinition {
        key: "fdn.kellan_planar_trailblazer",
        scryfall_id: "f46a9329-7b91-441d-8653-50c1152c9120",
        printed: PrintedCharacteristics {
            name: "Kellan, Planar Trailblazer".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Faerie".into(), "Scout".into()],
            power: Some(2),
            toughness: Some(1),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{1}{R}"),
                    effects: vec![Effect::If {
                        condition: Condition::SourceMatches(CardPredicate::subtype("Scout")),
                        then: vec![
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::SetCreatureTypes(vec![
                                    "Human".into(),
                                    "Faerie".into(),
                                    "Detective".into(),
                                ]),
                            ),
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::GrantTriggered(Box::new(Ability::Triggered {
                                    condition: TriggerCondition::DealsCombatDamage {
                                        subject: TriggerSubject::This,
                                        to: Some(Actor::AnyPlayer),
                                        per: Per::Object,
                                    },
                                    intervening_if: None,
                                    effects: vec![Effect::exile_top_to_play(
                                        1,
                                        PlayWindow::ThisTurn,
                                    )],
                                })),
                            ),
                        ],
                        otherwise: vec![],
                    }],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{2}{R}"),
                    effects: vec![Effect::If {
                        condition: Condition::SourceMatches(CardPredicate::subtype("Detective")),
                        then: vec![
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::SetCreatureTypes(vec![
                                    "Human".into(),
                                    "Faerie".into(),
                                    "Rogue".into(),
                                ]),
                            ),
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::base_pt(Some(3), Some(2)),
                            ),
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
                            ),
                        ],
                        otherwise: vec![],
                    }],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn drake_hatcher() -> CardDefinition {
    let text_box = "Vigilance, prowess\nWhenever this creature deals combat damage to a player, put that many incubation counters on it.\nRemove three incubation counters from this creature: Create a 2/2 blue Drake creature token with flying.";
    CardDefinition {
        key: "fdn.drake_hatcher",
        scryfall_id: "bcaf4196-6bf3-47fa-b5c7-0e77f45cf820",
        printed: PrintedCharacteristics {
            name: "Drake Hatcher".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::casts(
                        Actor::You,
                        CardPredicate {
                            not_card_types: vec![CardType::Creature],
                            ..Default::default()
                        },
                    ),
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::pump(1, 1),
                    )],
                },
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::DealsCombatDamage {
                        subject: TriggerSubject::This,
                        to: Some(Actor::AnyPlayer),
                        per: Per::Object,
                    },
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::Incubation,
                        EffectValue::TriggeringAmount,
                    )],
                },
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                remove_counters: Some((CounterKind::Incubation, 3)),
                effects: vec![Effect::create_tokens("token.drake", 1)],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fishing_pole() -> CardDefinition {
    let text_box = "Equipped creature has \"{1}, {T}, Tap Fishing Pole: Put a bait counter on Fishing Pole.\"\nWhenever equipped creature becomes untapped, remove a bait counter from this Equipment. If you do, create a 1/1 blue Fish creature token.\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "fdn.fishing_pole",
        scryfall_id: "c95ab836-3277-4223-9aaa-ef2c77256b65",
        printed: PrintedCharacteristics {
            name: "Fishing Pole".into(),
            mana_cost: Some(ManaCost::parse("{1}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            activated_abilities: vec![ActivatedAbilityDefinition {
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                }],
                ..Default::default()
            }],
            statics: vec![StaticAbility::AttachedHas(ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                tap_source: true,
                tap_grantor: true,
                effects: vec![Effect::put_counters(
                    ObjectSelector::Grantor,
                    CounterKind::Bait,
                    1,
                )],
                ..Default::default()
            })],
            // "Remove a bait counter. If you do, ...": with no effect that
            // replaces counter removal, removing one happens exactly when
            // the Pole still has one.
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::BecomesUntapped {
                    subject: TriggerSubject::Attached,
                },
                effects: vec![Effect::If {
                    condition: Condition::SourceCountersAtLeast {
                        kind: CounterKind::Bait,
                        count: 1,
                    },
                    then: vec![
                        Effect::RemoveCounters {
                            from: ObjectSelector::This,
                            up_to: 1,
                            chosen: Some(CounterKind::Bait),
                        },
                        Effect::create_tokens("token.fish", 1),
                    ],
                    otherwise: Vec::new(),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ravenous_amulet() -> CardDefinition {
    let text_box = "{1}, {T}, Sacrifice a creature: Draw a card and put a soul counter on this artifact. Activate only as a sorcery.\n{4}, {T}, Sacrifice this artifact: Each opponent loses life equal to the number of soul counters on this artifact.";
    CardDefinition {
        key: "fdn.ravenous_amulet",
        scryfall_id: "80cadee5-6f26-4440-ad31-a8e573a90436",
        printed: PrintedCharacteristics {
            name: "Ravenous Amulet".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{1}"),
                    tap_source: true,
                    sacrifice: Some(SacrificeCost {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        },
                        another: false,
                        or_mana: None,
                    }),
                    sorcery_timing: true,
                    effects: vec![
                        Effect::DrawCards {
                            count: EffectValue::Fixed(1),
                        },
                        Effect::put_counters(ObjectSelector::This, CounterKind::Soul, 1),
                    ],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{4}"),
                    tap_source: true,
                    sacrifice_source: true,
                    effects: vec![Effect::LoseLife {
                        player: PlayerSelector::EachOpponent,
                        amount: EffectValue::Property {
                            of: Box::new(ObjectSelector::This),
                            property: ObjectProperty::Counters(CounterKind::Soul),
                            aggregate: Aggregate::Sum,
                        },
                    }],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn felling_blow() -> CardDefinition {
    CardDefinition {
        key: "fdn.felling_blow",
        scryfall_id: "96948ae3-b15d-4d6d-aa73-9f52084cd903",
        printed: PrintedCharacteristics {
            name: "Felling Blow".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![
                TargetRequirement::one(TargetSpec::CreatureYouControl),
                TargetRequirement::one(TargetSpec::CreatureOpponentControls),
            ],
            spell_effects: vec![
                Effect::put_counters(ObjectSelector::Target(TargetSpec::CreatureYouControl), CounterKind::PlusOnePlusOne, 1),
                Effect::TargetCreaturesDealPowerDamageToLastTarget,
            ],
            text_box: "Put a +1/+1 counter on target creature you control. Then that creature deals damage equal to its power to target creature an opponent controls.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lunar_insight() -> CardDefinition {
    CardDefinition {
        key: "fdn.lunar_insight",
        scryfall_id: "a9a159f6-fecf-4bdd-b2f8-a9665a5cc32d",
        printed: PrintedCharacteristics {
            name: "Lunar Insight".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::DrawCards {
                count: EffectValue::DistinctManaValues {
                    of: Box::new(ObjectSelector::each(
                        Scope::YouControl,
                        CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    )),
                },
            }],
            text_box:
                "Draw a card for each different mana value among nonland permanents you control."
                    .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn luminous_rebuke() -> CardDefinition {
    let text_box = "This spell costs {3} less to cast if it targets a tapped creature.\nDestroy target creature.";
    CardDefinition {
        key: "fdn.luminous_rebuke",
        scryfall_id: "621839e1-2756-4cdc-a25c-5f76ea98dd87",
        printed: PrintedCharacteristics {
            name: "Luminous Rebuke".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Creature,
            ))],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            statics: vec![StaticAbility::CostsLessIfTargets(TargetCostReduction {
                predicate: CardPredicate {
                    tapped: true,
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                generic: 3,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn blasphemous_edict() -> CardDefinition {
    let text_box = "You may pay {B} rather than pay this spell's mana cost if there are thirteen or more creatures on the battlefield.\nEach player sacrifices thirteen creatures of their choice.";
    CardDefinition {
        key: "fdn.blasphemous_edict",
        scryfall_id: "11040ecd-3153-4029-b42b-1441bc51ec34",
        printed: PrintedCharacteristics {
            name: "Blasphemous Edict".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::AlternativeMana {
                cost: ManaCost::parse("{B}"),
                condition: Condition::BattlefieldAtLeast {
                    count: 13,
                    predicate: CardPredicate::creature(),
                },
            }],
            spell_effects: vec![Effect::EachPlayerSacrificesCreatures { count: 13 }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn claws_out() -> CardDefinition {
    let text_box = "Affinity for Cats (This spell costs {1} less to cast for each Cat you control.)\nCreatures you control get +2/+2 until end of turn.";
    CardDefinition {
        key: "fdn.claws_out",
        scryfall_id: "4396049c-b976-4b7f-8ecd-564e24ebd631",
        printed: PrintedCharacteristics {
            name: "Claws Out".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::CostsLessPerPermanent(
                CardPredicate::subtype("Cat"),
            )],
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::each_you_control(CardPredicate::creature()),
                Modification::pump(2, 2),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cat_collector() -> CardDefinition {
    let text_box = "When this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever you gain life for the first time during each of your turns, create a 1/1 white Cat creature token.";
    CardDefinition {
        key: "fdn.cat_collector",
        scryfall_id: "526fe356-bff1-4211-9e88-bf913ac76b1d",
        printed: PrintedCharacteristics {
            name: "Cat Collector".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Citizen".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::create_tokens("token.food", 1)],
                },
                Ability::Triggered {
                    condition: TriggerCondition::GainsLife {
                        who: Actor::You,
                        nth: Some(1),
                    },
                    intervening_if: Some(Condition::YourTurn),
                    effects: vec![Effect::create_tokens("token.cat_1_1_w", 1)],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fiendish_panda() -> CardDefinition {
    let text_box = "Whenever you gain life, put a +1/+1 counter on this creature.\nWhen this creature dies, return another target non-Bear creature card with mana value less than or equal to this creature's power from your graveyard to the battlefield.";
    let spec = TargetSpec::GraveyardCardYouOwnWithinSourcePower {
        predicate: CardPredicate {
            not_subtypes: vec!["Bear".into()],
            ..CardPredicate::creature()
        },
    };
    CardDefinition {
        key: "fdn.fiendish_panda",
        scryfall_id: "4e434d74-cad0-45f5-bc8d-f34aa5e1d879",
        printed: PrintedCharacteristics {
            name: "Fiendish Panda".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bear".into(), "Demon".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::GainsLife {
                        who: Actor::You,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::TargetClause {
                        requirement: TargetRequirement::another(spec.clone()),
                        effects: vec![Effect::return_to_battlefield(
                            ObjectSelector::Target(spec),
                            false,
                        )],
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cackling_prowler() -> CardDefinition {
    CardDefinition {
        key: "fdn.cackling_prowler",
        scryfall_id: "1bd8e971-c075-4203-8d83-c28f22d4f9b9",
        printed: PrintedCharacteristics {
            name: "Cackling Prowler".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Hyena".into(), "Rogue".into()],
            power: Some(4),
            toughness: Some(3),
            statics: vec![StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse("{2}")))],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 1)],
            }],
            text_box: "Ward {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)\nMorbid — At the beginning of your end step, if a creature died this turn, put a +1/+1 counter on this creature.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn high_society_hunter() -> CardDefinition {
    let text_box = "Flying\nWhenever this creature attacks, you may sacrifice another creature. If you do, put a +1/+1 counter on this creature.\nWhenever another nontoken creature dies, draw a card.";
    CardDefinition {
        key: "fdn.high_society_hunter",
        scryfall_id: "51da4a4b-ea12-4169-a7cf-eb4427f13e84",
        printed: PrintedCharacteristics {
            name: "High-Society Hunter".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Noble".into()],
            power: Some(5),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: text_box.into(),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                    intervening_if: None,
                    effects: vec![
                        Effect::may_sacrifice(CardPredicate::creature(), true),
                        Effect::if_you_do(vec![Effect::put_counters(
                            ObjectSelector::This,
                            CounterKind::PlusOnePlusOne,
                            1,
                        )]),
                    ],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::another(CardPredicate {
                            token: Some(false),
                            ..CardPredicate::creature()
                        }),
                    },
                    intervening_if: None,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn leyline_axe() -> CardDefinition {
    use managym_state::card::{StaticKeywordGrant, StaticPtBuff, StaticScope};
    CardDefinition {
        key: "fdn.leyline_axe",
        scryfall_id: "b9c03336-a321-4c06-94d1-809f328fabd8",
        printed: PrintedCharacteristics {
            name: "Leyline Axe".into(),
            mana_cost: Some(ManaCost::parse("{4}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{3}"))],
            statics: vec![StaticAbility::BeginOnBattlefield,
                StaticAbility::PtBuff(StaticPtBuff {
                    scope: StaticScope::Attached,
                    condition: None,
                    power: 1,
                    toughness: 1,
                    per_creature_type_max: None,
                    per_color_among: None,
                    per_graveyard_matching: None,
                }),
                StaticAbility::KeywordGrant(StaticKeywordGrant {
                    scope: StaticScope::Attached,
                    condition: None,
                    keywords: Keywords::of(&[Keyword::DoubleStrike, Keyword::Trample]),
                }),
            ],
            text_box: "If this card is in your opening hand, you may begin the game with it on the battlefield.\nEquipped creature gets +1/+1 and has double strike and trample.\nEquip {3}".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn incinerating_blast() -> CardDefinition {
    CardDefinition {
        key: "fdn.incinerating_blast",
        scryfall_id: "d58e20ab-c5ca-4295-884d-78efdaa83243",
        printed: PrintedCharacteristics {
            name: "Incinerating Blast".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![
                Effect::deal_damage(6, ObjectSelector::Target(TargetSpec::Creature)),
                Effect::discard_draw(1, 1),
            ],
            text_box: "Incinerating Blast deals 6 damage to target creature.\nYou may discard a card. If you do, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn on_second_draw(effect: Effect) -> Ability {
    Ability::Triggered {
        condition: TriggerCondition::Draws {
            who: Actor::You,
            nth: Some(2),
        },
        intervening_if: None,
        effects: vec![effect],
    }
}

fn erudite_wizard() -> CardDefinition {
    let text_box =
        "Whenever you draw your second card each turn, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "fdn.erudite_wizard",
        scryfall_id: "9273c417-0fcd-4273-b24e-afff76336d0c",
        printed: PrintedCharacteristics {
            name: "Erudite Wizard".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![on_second_draw(Effect::put_counters(
                ObjectSelector::This,
                CounterKind::PlusOnePlusOne,
                1,
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn homunculus_horde() -> CardDefinition {
    let text_box = "Whenever you draw your second card each turn, create a token that's a copy of this creature.";
    CardDefinition {
        key: "fdn.homunculus_horde",
        scryfall_id: "470c4d03-340a-4e0e-a59f-f19d05497785",
        printed: PrintedCharacteristics {
            name: "Homunculus Horde".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Homunculus".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![on_second_draw(Effect::CreateTokenCopy {
                of: ObjectSelector::This,
                exceptions: vec![],
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mischievous_mystic() -> CardDefinition {
    let text_box = "Flying\nWhenever you draw your second card each turn, create a 1/1 blue Faerie creature token with flying.";
    CardDefinition {
        key: "fdn.mischievous_mystic",
        scryfall_id: "20d89cec-528b-4b2a-87db-e11ce0000622",
        printed: PrintedCharacteristics {
            name: "Mischievous Mystic".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![on_second_draw(Effect::create_tokens(
                "token.faerie_1_1_u_flying",
                1,
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn chart_a_course() -> CardDefinition {
    CardDefinition {
        key: "fdn.chart_a_course",
        scryfall_id: "7599d459-fe71-48f4-8c65-0f519bb63a68",
        printed: PrintedCharacteristics {
            name: "Chart a Course".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                },
                Effect::If {
                    condition: Condition::AttackedThisTurn,
                    then: vec![],
                    otherwise: vec![Effect::discard()],
                },
            ],
            text_box: "Draw two cards. Then discard a card unless you attacked this turn.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn linden_the_steadfast_queen() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nWhenever a white creature you control attacks, you gain 1 life.";
    CardDefinition {
        key: "fdn.linden_the_steadfast_queen",
        scryfall_id: "e0bb07cd-9b74-4ba7-8089-27a565d74197",
        printed: PrintedCharacteristics {
            name: "Linden, the Steadfast Queen".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Noble".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate {
                        colors_any: vec![Color::White],
                        ..CardPredicate::creature()
                    }),
                    Per::Object,
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

fn herald_of_eternal_dawn() -> CardDefinition {
    CardDefinition {
        key: "fdn.herald_of_eternal_dawn",
        scryfall_id: "c9fdfebf-98e0-4718-bac3-6eee1cd0623d",
        printed: PrintedCharacteristics {
            name: "Herald of Eternal Dawn".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Flash]),
            statics: vec![StaticAbility::CantLoseAndOpponentsCantWin],
            text_box: "Flash (You may cast this spell any time you could cast an instant.)\nFlying\nYou can't lose the game and your opponents can't win the game.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn armasaur_guide() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nWhenever you attack with three or more creatures, put a +1/+1 counter on target creature you control.";
    CardDefinition {
        key: "fdn.armasaur_guide",
        scryfall_id: "c80fc380-0499-4499-8a60-c43844c02c9b",
        printed: PrintedCharacteristics {
            name: "Armasaur Guide".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                    per: Per::Batch,
                    attackers: AttackSize::AtLeast(3),
                    nth: None,
                },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::Target(TargetSpec::CreatureYouControl),
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn skyship_buccaneer() -> CardDefinition {
    let text_box =
        "Flying\nRaid — When this creature enters, if you attacked this turn, draw a card.";
    CardDefinition {
        key: "fdn.skyship_buccaneer",
        scryfall_id: "62958fc3-55dc-4b97-a070-490d6ed27820",
        printed: PrintedCharacteristics {
            name: "Skyship Buccaneer".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pirate".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::DrawCards { count: 1.into() }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn midnight_snack() -> CardDefinition {
    let text_box = "Raid — At the beginning of your end step, if you attacked this turn, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\n{2}{B}, Sacrifice this enchantment: Target opponent loses X life, where X is the amount of life you gained this turn.";
    CardDefinition {
        key: "fdn.midnight_snack",
        scryfall_id: "c9b7543f-2a45-4db6-b560-d15507a58c91",
        printed: PrintedCharacteristics {
            name: "Midnight Snack".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::create_tokens("token.food", 1)],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{B}"),
                sacrifice_source: true,
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                    amount: EffectValue::LifeGainedThisTurn(PlayerSelector::You),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn battlesong_berserker() -> CardDefinition {
    let text_box = "Whenever you attack, target creature you control gets +1/+0 and gains menace until end of turn. (It can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "fdn.battlesong_berserker",
        scryfall_id: "a1f8b199-5d62-485f-b1c3-b30aa550595b",
        printed: PrintedCharacteristics {
            name: "Battlesong Berserker".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Berserker".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Batch,
                ),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureYouControl),
                    effects: vec![
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::pump(1, 0),
                        ),
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                        ),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn goblin_boarders() -> CardDefinition {
    let text_box =
        "Raid — This creature enters with a +1/+1 counter on it if you attacked this turn.";
    CardDefinition {
        key: "fdn.goblin_boarders",
        scryfall_id: "4409a063-bf2a-4a49-803e-3ce6bd474353",
        printed: PrintedCharacteristics {
            name: "Goblin Boarders".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Pirate".into()],
            power: Some(3),
            toughness: Some(2),
            replacement_effects: vec![ReplacementEffect::EntersWithPlusOneCounters {
                count: 1,
                condition: Some(Condition::AttackedThisTurn),
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gorehorn_raider() -> CardDefinition {
    let text_box = "Raid — When this creature enters, if you attacked this turn, this creature deals 2 damage to any target.";
    CardDefinition {
        key: "fdn.gorehorn_raider",
        scryfall_id: "78ce6c40-3452-4aa0-a45b-dbfd70f8d220",
        printed: PrintedCharacteristics {
            name: "Gorehorn Raider".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Minotaur".into(), "Pirate".into()],
            power: Some(4),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::deal_damage(
                    2,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn searslicer_goblin() -> CardDefinition {
    let text_box = "Raid — At the beginning of your end step, if you attacked this turn, create a 1/1 red Goblin creature token.";
    CardDefinition {
        key: "fdn.searslicer_goblin",
        scryfall_id: "94ad0b97-a318-4e76-ac79-b3e83417c333",
        printed: PrintedCharacteristics {
            name: "Searslicer Goblin".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::create_tokens("token.goblin_1_1_r", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn alesha_who_laughs_at_fate() -> CardDefinition {
    let text_box = "First strike\nWhenever Alesha attacks, put a +1/+1 counter on it.\nRaid — At the beginning of your end step, if you attacked this turn, return target creature card with mana value less than or equal to Alesha's power from your graveyard to the battlefield.";
    CardDefinition {
        key: "fdn.alesha_who_laughs_at_fate",
        scryfall_id: "a93e3406-4e29-4bc0-ae52-cbd2ac1f99a4",
        printed: PrintedCharacteristics {
            name: "Alesha, Who Laughs at Fate".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                    intervening_if: Some(Condition::AttackedThisTurn),
                    effects: vec![Effect::return_to_battlefield(
                        ObjectSelector::Target(TargetSpec::GraveyardCardYouOwnWithinSourcePower {
                            predicate: CardPredicate::creature(),
                        }),
                        false,
                    )],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ashroot_animist() -> CardDefinition {
    let text_box = "Trample\nWhenever this creature attacks, another target creature you control gains trample and gets +X/+X until end of turn, where X is this creature's power.";
    CardDefinition {
        key: "fdn.ashroot_animist",
        scryfall_id: "dece147f-d71a-4c95-9fe5-f5c862ef14ac",
        printed: PrintedCharacteristics {
            name: "Ashroot Animist".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Lizard".into(), "Druid".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::another(TargetSpec::CreatureYouControl),
                    effects: vec![
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                        ),
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::ModifyPt {
                                power: EffectValue::SourcePower,
                                toughness: EffectValue::SourcePower,
                            },
                        ),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bulk_up() -> CardDefinition {
    let text_box = "Double target creature's power until end of turn.\nFlashback {4}{R}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "fdn.bulk_up",
        scryfall_id: "977dcc50-da10-4281-b522-9240c1204f5d",
        printed: PrintedCharacteristics {
            name: "Bulk Up".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{4}{R}{R}"))],
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::ModifyPt {
                    power: ObjectSelector::Target(TargetSpec::Creature)
                        .property(ObjectProperty::Power, Aggregate::Sum),
                    toughness: 0.into(),
                },
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn courageous_goblin() -> CardDefinition {
    let text_box = "Whenever this creature attacks while you control a creature with power 4 or greater, this creature gets +1/+0 and gains menace until end of turn. (It can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "fdn.courageous_goblin",
        scryfall_id: "8db6819c-666a-409d-85a5-b9ac34d8dd2f",
        printed: PrintedCharacteristics {
            name: "Courageous Goblin".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::compare(
                        ObjectSelector::each_you_control(CardPredicate {
                            min_power: Some(4),
                            ..CardPredicate::creature()
                        })
                        .count(),
                        CompareOp::Ge,
                        1,
                    ),
                    condition: Box::new(TriggerCondition::attacks(
                        TriggerSubject::This,
                        Per::Object,
                    )),
                },
                intervening_if: None,
                effects: vec![
                    Effect::until_end_of_turn(ObjectSelector::This, Modification::pump(1, 0)),
                    Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tragic_banshee() -> CardDefinition {
    let text_box = "Morbid — When this creature enters, target creature an opponent controls gets -1/-1 until end of turn. If a creature died this turn, that creature gets -13/-13 until end of turn instead.";
    let victim = || ObjectSelector::Target(TargetSpec::CreatureOpponentControls);
    CardDefinition {
        key: "fdn.tragic_banshee",
        scryfall_id: "30df3e33-2f17-4067-99f1-5db6b0f41fd4",
        printed: PrintedCharacteristics {
            name: "Tragic Banshee".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into()],
            power: Some(5),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                // Death history is read as the ability resolves (CR 608.2h).
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureOpponentControls),
                    effects: vec![Effect::If {
                        condition: Condition::CreatureDiedThisTurn,
                        then: vec![Effect::until_end_of_turn(
                            victim(),
                            Modification::pump(-13, -13),
                        )],
                        otherwise: vec![Effect::until_end_of_turn(
                            victim(),
                            Modification::pump(-1, -1),
                        )],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vengeful_bloodwitch() -> CardDefinition {
    let text_box = "Whenever this creature or another creature you control dies, target opponent loses 1 life and you gain 1 life.";
    CardDefinition {
        key: "fdn.vengeful_bloodwitch",
        scryfall_id: "bd0c12dd-f138-45c0-9614-d83a1d8e8399",
        printed: PrintedCharacteristics {
            name: "Vengeful Bloodwitch".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Warlock".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Opponent),
                    effects: vec![
                        Effect::LoseLife {
                            player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                            amount: EffectValue::Fixed(1),
                        },
                        Effect::gain_life(1),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn needletooth_pack() -> CardDefinition {
    let text_box = "Morbid — At the beginning of your end step, if a creature died this turn, put two +1/+1 counters on target creature you control.";
    CardDefinition {
        key: "fdn.needletooth_pack",
        scryfall_id: "993c1679-e02b-44f2-b34e-12fd6b5142e9",
        printed: PrintedCharacteristics {
            name: "Needletooth Pack".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(4),
            toughness: Some(5),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureYouControl),
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        2,
                    )],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn slumbering_cerberus() -> CardDefinition {
    let text_box = "This creature doesn't untap during your untap step.\nMorbid — At the beginning of each end step, if a creature died this turn, untap this creature.";
    CardDefinition {
        key: "fdn.slumbering_cerberus",
        scryfall_id: "9d06faa8-201d-45db-b398-ad56f7b01848",
        printed: PrintedCharacteristics {
            name: "Slumbering Cerberus".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dog".into()],
            power: Some(4),
            toughness: Some(2),
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::This,
                kind: Restriction::DoesntUntap,
                unless: None,
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::untap(ObjectSelector::This)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wardens_of_the_cycle() -> CardDefinition {
    let text_box = "Morbid — At the beginning of your end step, if a creature died this turn, choose one —\n• You gain 2 life.\n• You draw a card and you lose 1 life.";
    CardDefinition {
        key: "fdn.wardens_of_the_cycle",
        scryfall_id: "83ea9b2c-5723-4eff-88ac-6669975939e3",
        printed: PrintedCharacteristics {
            name: "Wardens of the Cycle".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warlock".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::TriggerModes {
                    modes: vec![
                        vec![Effect::gain_life(2)],
                        vec![
                            Effect::DrawCards {
                                count: EffectValue::Fixed(1),
                            },
                            Effect::LoseLife {
                                player: PlayerSelector::You,
                                amount: EffectValue::Fixed(1),
                            },
                        ],
                    ],
                    up_to: None,
                    unchosen: false,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn scrawling_crawler() -> CardDefinition {
    let text_box = "At the beginning of your upkeep, each player draws a card.\nWhenever an opponent draws a card, that player loses 1 life.";
    CardDefinition {
        key: "fdn.scrawling_crawler",
        scryfall_id: "a1176dcf-40ee-4342-aa74-791b8352e99a",
        printed: PrintedCharacteristics {
            name: "Scrawling Crawler".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Phyrexian".into(), "Construct".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                    intervening_if: None,
                    // Each player draws in APNAP order (CR 101.4).
                    effects: vec![Effect::ForPlayer {
                        player: PlayerSelector::EachPlayer,
                        effects: vec![Effect::DrawCards {
                            count: EffectValue::Fixed(1),
                        }],
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Draws {
                        who: Actor::Opponent,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::LoseLife {
                        player: PlayerSelector::TriggeringPlayer,
                        amount: EffectValue::Fixed(1),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn firespitter_whelp() -> CardDefinition {
    let text_box = "Flying\nWhenever you cast a noncreature or Dragon spell, this creature deals 1 damage to each opponent.";
    CardDefinition {
        key: "fdn.firespitter_whelp",
        scryfall_id: "4b3a4c7d-3126-4bde-9dca-cb6a1e2f37c9",
        printed: PrintedCharacteristics {
            name: "Firespitter Whelp".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                // One predicate: a noncreature Dragon spell triggers it once.
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        any_of: vec![
                            CardPredicate {
                                not_card_types: vec![CardType::Creature],
                                ..Default::default()
                            },
                            CardPredicate::subtype("Dragon"),
                        ],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn archmage_of_runes() -> CardDefinition {
    let text_box = "Instant and sorcery spells you cast cost {1} less to cast.\nWhenever you cast an instant or sorcery spell, draw a card.";
    let instant_or_sorcery = || CardPredicate {
        card_types_any: vec![CardType::Instant, CardType::Sorcery],
        ..Default::default()
    };
    CardDefinition {
        key: "fdn.archmage_of_runes",
        scryfall_id: "334b5018-2da9-49f1-9d09-83d312ecfb02",
        printed: PrintedCharacteristics {
            name: "Archmage of Runes".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Giant".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(6),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                first_on_own_turn: false,
                predicate: instant_or_sorcery(),
                condition: Condition::Always,
                generic: 1.into(),
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, instant_or_sorcery()),
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn goblin_negotiation() -> CardDefinition {
    let text_box = "Goblin Negotiation deals X damage to target creature. Create a number of 1/1 red Goblin creature tokens equal to the amount of excess damage dealt to that creature this way.";
    CardDefinition {
        key: "fdn.goblin_negotiation",
        scryfall_id: "f2016585-e26c-4d13-b09f-af6383c192f7",
        printed: PrintedCharacteristics {
            name: "Goblin Negotiation".into(),
            mana_cost: Some(ManaCost::parse("{X}{R}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![
                Effect::deal_damage(EffectValue::X, ObjectSelector::Target(TargetSpec::Creature)),
                // Excess is measured as the damage is dealt (CR 120.4a):
                // prevented damage leaves none.
                Effect::create_tokens("token.goblin_1_1_r", EffectValue::ExcessDamage),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn niv_mizzet_visionary() -> CardDefinition {
    let text_box = "Flying\nYou have no maximum hand size.\nWhenever a source you control deals noncombat damage to an opponent, you draw that many cards.";
    CardDefinition {
        key: "fdn.niv_mizzet_visionary",
        scryfall_id: "7a69a618-d588-4745-8ede-0ff0a9f356f1",
        printed: PrintedCharacteristics {
            name: "Niv-Mizzet, Visionary".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into(), "Wizard".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::NoMaximumHandSize],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::DealsDamage {
                    kind: DamageKind::Noncombat,
                    source: TriggerSubject::you_control(CardPredicate::default()),
                    to: DamageRecipientFilter::Player(Actor::Opponent),
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::TriggeringAmount,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hare_apparent() -> CardDefinition {
    let text_box = "When this creature enters, create a number of 1/1 white Rabbit creature tokens equal to the number of other creatures you control named Hare Apparent.\nA deck can have any number of cards named Hare Apparent.";
    // CR 201.5c: the count reads other creatures by name as the trigger
    // resolves, whatever happened to this one.
    let namesakes = CardPredicate {
        card_type: Some(CardType::Creature),
        named: Some("Hare Apparent".into()),
        ..Default::default()
    };
    CardDefinition {
        key: "fdn.hare_apparent",
        scryfall_id: "9fc6f0e9-eb5f-4bc0-b3d7-756644b66d12",
        printed: PrintedCharacteristics {
            name: "Hare Apparent".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rabbit".into(), "Noble".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::create_tokens(
                    "token.rabbit",
                    ObjectSelector::each_other_you_control(namesakes).count(),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ingenious_leonin() -> CardDefinition {
    let text_box = "{3}{W}: Put a +1/+1 counter on another target attacking creature you control. If that creature is a Cat, it gains first strike until end of turn. (It deals combat damage before creatures without first strike.)";
    let spec = TargetSpec::PermanentYouControl {
        predicate: CardPredicate {
            card_type: Some(CardType::Creature),
            attacking: true,
            ..Default::default()
        },
    };
    let target = || ObjectSelector::Target(spec.clone());
    CardDefinition {
        key: "fdn.ingenious_leonin",
        scryfall_id: "ea566679-4202-4076-9314-142241485f6e",
        printed: PrintedCharacteristics {
            name: "Ingenious Leonin".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Soldier".into()],
            power: Some(4),
            toughness: Some(4),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{W}"),
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::another(spec.clone()),
                    effects: vec![
                        Effect::put_counters(target(), CounterKind::PlusOnePlusOne, 1),
                        Effect::If {
                            condition: Condition::TargetMatches(CardPredicate::subtype("Cat")),
                            then: vec![Effect::until_end_of_turn(
                                target(),
                                Modification::AddKeywords(Keywords::of(&[Keyword::FirstStrike])),
                            )],
                            otherwise: vec![],
                        },
                    ],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fleeting_flight() -> CardDefinition {
    let text_box = "Put a +1/+1 counter on target creature. It gains flying until end of turn. Prevent all combat damage that would be dealt to it this turn.";
    let target = || ObjectSelector::Target(TargetSpec::Creature);
    CardDefinition {
        key: "fdn.fleeting_flight",
        scryfall_id: "55139100-9342-41fd-b10a-8e9932e605d4",
        printed: PrintedCharacteristics {
            name: "Fleeting Flight".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::put_counters(target(), CounterKind::PlusOnePlusOne, 1),
                Effect::until_end_of_turn(
                    target(),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
                ),
                Effect::until_end_of_turn(target(), Modification::PreventCombatDamage),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn grappling_kraken() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, tap target creature an opponent controls and put a stun counter on it. (If a permanent with a stun counter would become untapped, remove one from it instead.)";
    let target = || ObjectSelector::Target(TargetSpec::CreatureOpponentControls);
    CardDefinition {
        key: "fdn.grappling_kraken",
        scryfall_id: "d1f5cab3-3fc0-448d-8252-cd55abf5b596",
        printed: PrintedCharacteristics {
            name: "Grappling Kraken".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Kraken".into()],
            power: Some(5),
            toughness: Some(6),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![
                    Effect::tap(target()),
                    Effect::put_counters(target(), CounterKind::Stun, 1),
                ],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn twinflame_tyrant() -> CardDefinition {
    CardDefinition {
        key: "fdn.twinflame_tyrant",
        scryfall_id: "1eb34f51-0bd2-43c3-af95-2ce8dabcc7bb",
        printed: PrintedCharacteristics {
            name: "Twinflame Tyrant".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            replacement_effects: vec![ReplacementEffect::ModifyDamage {
                creature_source: false,
                source_by: Actor::You,
                recipient_by: Actor::Opponent,
                transform: DamageTransform::Double,
            }],
            text_box: "Flying\nIf a source you control would deal damage to an opponent or a permanent an opponent controls, it deals double that damage instead.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lightshell_duo() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nWhen this creature enters, surveil 2. (Look at the top two cards of your library, then put any number of them into your graveyard and the rest on top of your library in any order.)";
    CardDefinition {
        key: "fdn.lightshell_duo",
        scryfall_id: "bb75315c-ea8f-4eb0-899e-c73ef75fc396",
        printed: PrintedCharacteristics {
            name: "Lightshell Duo".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rat".into(), "Otter".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![
                Ability::prowess(),
                Ability::when_enters(TriggerSubject::This, vec![Effect::Surveil { count: 2 }]),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn arahbo_the_first_fang() -> CardDefinition {
    let cat = CardPredicate::subtype("Cat");
    let other_nontoken_cat = TriggerSubject::another_you_control(CardPredicate {
        token: Some(false),
        ..cat.clone()
    });
    let text_box = "Other Cats you control get +1/+1.\nWhenever Arahbo or another nontoken Cat you control enters, create a 1/1 white Cat creature token.";
    CardDefinition {
        key: "fdn.arahbo_the_first_fang",
        scryfall_id: "524a5d93-26ed-436d-a437-dc9460acce98",
        printed: PrintedCharacteristics {
            name: "Arahbo, the First Fang".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Avatar".into()],
            power: Some(2),
            toughness: Some(2),
            supertypes: vec!["Legendary".into()],
            statics: vec![StaticAbility::continuous(
                StaticScope::OtherYouControl(cat),
                None,
                vec![Modification::pump(1, 1)],
            )],
            abilities: vec![Ability::when_enters(
                TriggerSubject::ThisOr(Box::new(other_nontoken_cat)),
                vec![Effect::create_tokens("token.cat_1_1_w", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn helpful_hunter() -> CardDefinition {
    CardDefinition {
        key: "fdn.helpful_hunter",
        scryfall_id: "1b9a0e91-80b5-428f-8f08-931d0631be14",
        printed: PrintedCharacteristics {
            name: "Helpful Hunter".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            )],
            text_box: "When this creature enters, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn prideful_parent() -> CardDefinition {
    CardDefinition {
        key: "fdn.prideful_parent",
        scryfall_id: "b742117a-8a72-43b9-b05d-274829d138a2",
        printed: PrintedCharacteristics {
            name: "Prideful Parent".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.cat_1_1_w", 1)],
            )],
            text_box: "Vigilance (Attacking doesn't cause this creature to tap.)\nWhen this creature enters, create a 1/1 white Cat creature token.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn anthem_of_champions() -> CardDefinition {
    CardDefinition {
        key: "fdn.anthem_of_champions",
        scryfall_id: "42fe3a40-9cbe-4235-86f9-32576aaebba8",
        printed: PrintedCharacteristics {
            name: "Anthem of Champions".into(),
            mana_cost: Some(ManaCost::parse("{G}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::continuous(
                StaticScope::YouControl(CardPredicate::creature()),
                None,
                vec![Modification::pump(1, 1)],
            )],
            text_box: "Creatures you control get +1/+1.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn guarded_heir() -> CardDefinition {
    let text_box = "Lifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhen this creature enters, create two 3/3 white Knight creature tokens.";
    CardDefinition {
        key: "fdn.guarded_heir",
        scryfall_id: "525ba5c7-3ce5-4e52-b8b5-96c9040a6738",
        printed: PrintedCharacteristics {
            name: "Guarded Heir".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Noble".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.knight_3_3_w", 2)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dragon_trainer() -> CardDefinition {
    let text_box = "When this creature enters, create a 4/4 red Dragon creature token with flying.";
    CardDefinition {
        key: "fdn.dragon_trainer",
        scryfall_id: "91bd75a1-cb54-4e38-9ce1-e8f32a73c6eb",
        printed: PrintedCharacteristics {
            name: "Dragon Trainer".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.dragon_4_4_r_flying", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rite_of_the_dragoncaller() -> CardDefinition {
    let text_box = "Whenever you cast an instant or sorcery spell, create a 5/5 red Dragon creature token with flying.";
    CardDefinition {
        key: "fdn.rite_of_the_dragoncaller",
        scryfall_id: "673e4561-8dfd-46db-b492-878009666ac7",
        printed: PrintedCharacteristics {
            name: "Rite of the Dragoncaller".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(Actor::You, CardPredicate::instant_or_sorcery()),
                effects: vec![Effect::create_tokens("token.dragon_5_5_r_flying", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elfsworn_giant() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)\nLandfall — Whenever a land you control enters, create a 1/1 green Elf Warrior creature token.";
    CardDefinition {
        key: "fdn.elfsworn_giant",
        scryfall_id: "5128a5be-ffa6-4998-8488-872d80b24cb2",
        printed: PrintedCharacteristics {
            name: "Elfsworn Giant".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Giant".into()],
            power: Some(5),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Reach]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Effect::create_tokens("token.elf_warrior_1_1_g", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn infestation_sage() -> CardDefinition {
    let text_box =
        "When this creature dies, create a 1/1 black and green Insect creature token with flying.";
    CardDefinition {
        key: "fdn.infestation_sage",
        scryfall_id: "d40c73de-7a5f-46f2-a70b-449bc8ecfe24",
        printed: PrintedCharacteristics {
            name: "Infestation Sage".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warlock".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::create_tokens("token.insect_1_1_bg_flying", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn eager_trufflesnout() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nWhenever this creature deals combat damage to a player, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";
    CardDefinition {
        key: "fdn.eager_trufflesnout",
        scryfall_id: "a6e8433d-eb2a-43d1-b59b-7d70ff97c8e7",
        printed: PrintedCharacteristics {
            name: "Eager Trufflesnout".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Boar".into()],
            power: Some(4),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Some(Actor::AnyPlayer),
                    per: Per::Object,
                },
                effects: vec![Effect::create_tokens("token.food", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn quakestrider_ceratops() -> CardDefinition {
    CardDefinition {
        key: "fdn.quakestrider_ceratops",
        scryfall_id: "067f72c2-ead6-4879-bc9d-696c9f87c0b2",
        printed: PrintedCharacteristics {
            name: "Quakestrider Ceratops".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(12),
            toughness: Some(8),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn stab() -> CardDefinition {
    CardDefinition {
        key: "fdn.stab",
        scryfall_id: "6859a5ba-1c1c-4631-bba8-f9900b827178",
        printed: PrintedCharacteristics {
            name: "Stab".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::pump(-2, -2),
            )],
            text_box: "Target creature gets -2/-2 until end of turn.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn goblin_surprise() -> CardDefinition {
    let text_box = "Choose one —\n• Creatures you control get +2/+0 until end of turn.\n• Create two 1/1 red Goblin creature tokens.";
    CardDefinition {
        key: "fdn.goblin_surprise",
        scryfall_id: "527dd5d4-5f72-40bb-8a9d-1f5ac3f81e2e",
        printed: PrintedCharacteristics {
            name: "Goblin Surprise".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                        Modification::pump(2, 0),
                    )],
                    vec![Effect::create_tokens("token.goblin_1_1_r", 2)],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn seekers_folly() -> CardDefinition {
    let text_box = "Choose one —\n• Target opponent discards two cards.\n• Creatures your opponents control get -1/-1 until end of turn.";
    CardDefinition {
        key: "fdn.seekers_folly",
        scryfall_id: "bc359da6-8b7f-45ec-b530-ce159fc35953",
        printed: PrintedCharacteristics {
            name: "Seeker's Folly".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::Discard {
                        what: ObjectSelector::Target(TargetSpec::Opponent),
                        count: 2,
                    }],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::OpponentsControl, CardPredicate::creature()),
                        Modification::pump(-1, -1),
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn clinquant_skymage() -> CardDefinition {
    CardDefinition {
        key: "fdn.clinquant_skymage",
        scryfall_id: "36012810-0e83-4640-8ba7-7262229f1b84",
        printed: PrintedCharacteristics {
            name: "Clinquant Skymage".into(),
            subtypes: vec!["Bird".into(), "Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Draws {
                    who: Actor::You,
                    nth: None,
                },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: "Flying\nWhenever you draw a card, put a +1/+1 counter on this creature."
                .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elvish_regrower() -> CardDefinition {
    let text_box =
        "When this creature enters, return target permanent card from your graveyard to your hand.";
    CardDefinition {
        key: "fdn.elvish_regrower",
        scryfall_id: "2694e3cd-26ed-4a10-ae55-fb84d7800253",
        printed: PrintedCharacteristics {
            name: "Elvish Regrower".into(),
            subtypes: vec!["Elf".into(), "Druid".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            power: Some(4),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::return_to_hand(ObjectSelector::Target(
                    TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate::permanent(),
                    },
                ))],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn twinblade_blessing() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nEnchant creature\nEnchanted creature has double strike. (It deals both first-strike and regular combat damage.)";
    CardDefinition {
        key: "fdn.twinblade_blessing",
        scryfall_id: "ecf01cbe-9fcb-4f35-bc6b-2280620b06ff",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::DoubleStrike,
                ]))],
            )],
            spell_effects: vec![Effect::enchant(TargetSpec::Creature)],
            name: "Twinblade Blessing".into(),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            mana_cost: Some(ManaCost::parse("{1}{W}{W}")),
            keywords: Keywords::of(&[Keyword::Flash]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn quick_draw_katana() -> CardDefinition {
    let text_box = "During your turn, equipped creature gets +2/+0 and has first strike. (It deals combat damage before creatures without first strike.)\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "fdn.quick_draw_katana",
        scryfall_id: "69beec98-c89c-4673-953c-8b3ef3d81560",
        printed: PrintedCharacteristics {
            name: "Quick-Draw Katana".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                Some(Condition::YourTurn),
                vec![
                    Modification::pump(2, 0),
                    Modification::AddKeywords(Keywords::of(&[Keyword::FirstStrike])),
                ],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn billowing_shriekmass() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, mill three cards. (Put the top three cards of your library into your graveyard.)\nThreshold — This creature gets +2/+1 as long as there are seven or more cards in your graveyard.";
    CardDefinition {
        key: "fdn.billowing_shriekmass",
        scryfall_id: "7b3587a9-0667-4d53-807b-c437bcb1d7b3",
        printed: PrintedCharacteristics {
            name: "Billowing Shriekmass".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::MillCards {
                    count: 3.into(),
                    target: None,
                    player: None,
                }],
            )],
            statics: vec![StaticAbility::continuous(
                StaticScope::This,
                Some(Condition::GraveyardAtLeast {
                    count: 7,
                    predicate: CardPredicate::default(),
                }),
                vec![Modification::pump(2, 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn inspiring_paladin() -> CardDefinition {
    let text_box = "During your turn, this creature has first strike. (It deals combat damage before creatures without first strike.)\nDuring your turn, creatures you control with +1/+1 counters on them have first strike.";
    CardDefinition {
        key: "fdn.inspiring_paladin",
        scryfall_id: "0763be06-25b2-4d6b-ab33-a1af85aeb443",
        printed: PrintedCharacteristics {
            name: "Inspiring Paladin".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Knight".into()],
            power: Some(3),
            toughness: Some(3),
            statics: vec![
                StaticAbility::continuous(
                    StaticScope::This,
                    Some(Condition::YourTurn),
                    vec![Modification::AddKeywords(Keywords::of(&[
                        Keyword::FirstStrike,
                    ]))],
                ),
                StaticAbility::continuous(
                    StaticScope::YouControl(CardPredicate {
                        with_counter: true,
                        ..CardPredicate::creature()
                    }),
                    Some(Condition::YourTurn),
                    vec![Modification::AddKeywords(Keywords::of(&[
                        Keyword::FirstStrike,
                    ]))],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn skyknight_squire() -> CardDefinition {
    let text_box = "Whenever another creature you control enters, put a +1/+1 counter on this creature.\nAs long as this creature has three or more +1/+1 counters on it, it has flying and is a Knight in addition to its other types.";
    CardDefinition {
        key: "fdn.skyknight_squire",
        scryfall_id: "fcfe4e62-c153-47b8-8e09-cedaf91f53d8",
        printed: PrintedCharacteristics {
            name: "Skyknight Squire".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Scout".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate::creature()),
                vec![Effect::put_counters(
                    ObjectSelector::ExactSource,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            )],
            statics: vec![StaticAbility::continuous(
                StaticScope::This,
                Some(Condition::SourceCountersAtLeast {
                    kind: CounterKind::PlusOnePlusOne,
                    count: 3,
                }),
                vec![
                    Modification::AddSubtypes(vec!["Knight".into()]),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
                ],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn starlight_snare() -> CardDefinition {
    let text_box = "Enchant creature\nWhen this Aura enters, tap enchanted creature.\nEnchanted creature doesn't untap during its controller's untap step.";
    CardDefinition {
        key: "fdn.starlight_snare",
        scryfall_id: "74fb19b2-4f6c-4cbd-8756-a7eb5c7c9ef6",
        printed: PrintedCharacteristics {
            name: "Starlight Snare".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            spell_effects: vec![Effect::enchant(TargetSpec::Creature)],
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::Attached,
                kind: Restriction::DoesntUntap,
                unless: None,
            })],
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::tap(ObjectSelector::Attached)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rune_sealed_wall() -> CardDefinition {
    let text_box = "Defender\n{T}: Surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "fdn.rune_sealed_wall",
        scryfall_id: "da0f147b-95ed-4f32-9b46-6a633ae31976",
        printed: PrintedCharacteristics {
            name: "Rune-Sealed Wall".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Wall".into()],
            power: Some(0),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Defender]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::Surveil { count: 1 }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn icewind_elemental() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, draw a card, then discard a card.";
    CardDefinition {
        key: "fdn.icewind_elemental",
        scryfall_id: "fd0eba76-3829-408b-828f-0b223c884728",
        printed: PrintedCharacteristics {
            name: "Icewind Elemental".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elemental".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::DrawCards { count: 1.into() }, Effect::discard()],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn strix_lookout() -> CardDefinition {
    let text_box = "Flying, vigilance (Attacking doesn't cause this creature to tap.)\n{1}{U}, {T}: Draw a card, then discard a card.";
    CardDefinition {
        key: "fdn.strix_lookout",
        scryfall_id: "fbd2422e-8e84-4c39-af29-3b4d38baee63",
        printed: PrintedCharacteristics {
            name: "Strix Lookout".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(1),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{U}"),
                tap_source: true,
                effects: vec![Effect::DrawCards { count: 1.into() }, Effect::discard()],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn boltwave() -> CardDefinition {
    let text_box = "Boltwave deals 3 damage to each opponent.";
    CardDefinition {
        key: "fdn.boltwave",
        scryfall_id: "8d1ec351-5e70-4eb2-b590-6bff94ef8178",
        printed: PrintedCharacteristics {
            name: "Boltwave".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::DealDamageToPlayer {
                player: PlayerSelector::EachOpponent,
                amount: 3.into(),
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn faebloom_trick() -> CardDefinition {
    let text_box = "Create two 1/1 blue Faerie creature tokens with flying. When you do, tap target creature an opponent controls.";
    CardDefinition {
        key: "fdn.faebloom_trick",
        scryfall_id: "0c3bee8f-f5be-4404-a696-c902637799c3",
        printed: PrintedCharacteristics {
            name: "Faebloom Trick".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::create_tokens("token.faerie_1_1_u_flying", 2),
                Effect::WhenYouDo {
                    effects: vec![Effect::tap(ObjectSelector::Target(
                        TargetSpec::CreatureOpponentControls,
                    ))],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn joust_through() -> CardDefinition {
    let text_box =
        "Joust Through deals 3 damage to target attacking or blocking creature. You gain 1 life.";
    CardDefinition {
        key: "fdn.joust_through",
        scryfall_id: "846adb38-f9bb-4fed-b8ed-36ec7885f989",
        printed: PrintedCharacteristics {
            name: "Joust Through".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::deal_damage(
                    3,
                    ObjectSelector::Target(TargetSpec::Permanent {
                        predicate: CardPredicate {
                            in_combat: true,
                            ..CardPredicate::creature()
                        },
                    }),
                ),
                Effect::gain_life(1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn preposterous_proportions() -> CardDefinition {
    let text_box = "Creatures you control get +10/+10 and gain vigilance until end of turn.";
    CardDefinition {
        key: "fdn.preposterous_proportions",
        scryfall_id: "acb65189-60e4-42e0-9fb1-da6b716b91d7",
        printed: PrintedCharacteristics {
            name: "Preposterous Proportions".into(),
            mana_cost: Some(ManaCost::parse("{5}{G}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::each_you_control(CardPredicate::creature()),
                10,
                10,
                Keyword::Vigilance,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn refute() -> CardDefinition {
    let text_box = "Counter target spell. Draw a card, then discard a card.";
    CardDefinition {
        key: "fdn.refute",
        scryfall_id: "38806934-dd9c-4ad4-a59c-a16dce03a14a",
        printed: PrintedCharacteristics {
            name: "Refute".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::CounterSpell {
                    what: ObjectSelector::Target(TargetSpec::Spell),
                    exile: false,
                },
                Effect::DrawCards { count: 1.into() },
                Effect::discard(),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn uncharted_voyage() -> CardDefinition {
    let text_box = "Target creature's owner puts it on their choice of the top or bottom of their library.\nSurveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "fdn.uncharted_voyage",
        scryfall_id: "e0846820-e595-4743-8a28-29c57d728677",
        printed: PrintedCharacteristics {
            name: "Uncharted Voyage".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::OwnerPutsIntoLibrary {
                    target: TargetSpec::Creature,
                    top: true,
                },
                Effect::Surveil { count: 1 },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ambush_wolf() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhen this creature enters, exile up to one target card from a graveyard.";
    CardDefinition {
        key: "fdn.ambush_wolf",
        scryfall_id: "2903832c-318e-42ab-bf58-c682ec2f7afd",
        printed: PrintedCharacteristics {
            name: "Ambush Wolf".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Wolf".into()],
            power: Some(4),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::TargetClause {
                    requirement: TargetRequirement::up_to(
                        1,
                        TargetSpec::GraveyardCard {
                            predicate: CardPredicate::default(),
                        },
                    ),
                    effects: vec![Effect::exile(ObjectSelector::Target(
                        TargetSpec::GraveyardCard {
                            predicate: CardPredicate::default(),
                        },
                    ))],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn beast_kin_ranger() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nWhenever another creature you control enters, this creature gets +1/+0 until end of turn.";
    CardDefinition {
        key: "fdn.beast_kin_ranger",
        scryfall_id: "0102e0be-5783-4825-9489-713b1b1df0b2",
        printed: PrintedCharacteristics {
            name: "Beast-Kin Ranger".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Ranger".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate::creature()),
                vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(1, 0),
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bigfin_bouncer() -> CardDefinition {
    let text_box = "When this creature enters, return target creature an opponent controls to its owner's hand.";
    CardDefinition {
        key: "fdn.bigfin_bouncer",
        scryfall_id: "9b1d5b76-b07e-45c6-800d-4cfce085164f",
        printed: PrintedCharacteristics {
            name: "Bigfin Bouncer".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Shark".into(), "Pirate".into()],
            power: Some(3),
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

fn crackling_cyclops() -> CardDefinition {
    let text_box =
        "Whenever you cast a noncreature spell, this creature gets +3/+0 until end of turn.";
    CardDefinition {
        key: "fdn.crackling_cyclops",
        scryfall_id: "6e5b899a-52f7-471b-ad50-4fa6566758fd",
        printed: PrintedCharacteristics {
            name: "Crackling Cyclops".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cyclops".into(), "Wizard".into()],
            power: Some(0),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(Actor::You, CardPredicate::noncreature()),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(3, 0),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dauntless_veteran() -> CardDefinition {
    let text_box =
        "Whenever this creature attacks, creatures you control get +1/+1 until end of turn.";
    CardDefinition {
        key: "fdn.dauntless_veteran",
        scryfall_id: "7a136f26-ac66-407f-b389-357222d2c4a2",
        printed: PrintedCharacteristics {
            name: "Dauntless Veteran".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::pump(1, 1),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dazzling_angel() -> CardDefinition {
    let text_box = "Flying\nWhenever another creature you control enters, you gain 1 life.";
    CardDefinition {
        key: "fdn.dazzling_angel",
        scryfall_id: "027dc444-e544-4693-8653-3dcdda530162",
        printed: PrintedCharacteristics {
            name: "Dazzling Angel".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate::creature()),
                vec![Effect::gain_life(1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elementalist_adept() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";
    CardDefinition {
        key: "fdn.elementalist_adept",
        scryfall_id: "d9768cc6-8f53-4922-ae32-376a2f32d719",
        printed: PrintedCharacteristics {
            name: "Elementalist Adept".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::prowess()],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn felidar_savior() -> CardDefinition {
    let text_box = "Lifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhen this creature enters, put a +1/+1 counter on each of up to two other target creatures you control.";
    CardDefinition {
        key: "fdn.felidar_savior",
        scryfall_id: "cd092b14-d72f-4de0-8f19-1338661b9e3b",
        printed: PrintedCharacteristics {
            name: "Felidar Savior".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Beast".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        another: true,
                        ..TargetRequirement::up_to(2, TargetSpec::CreatureYouControl)
                    },
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Targets(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hinterland_sanctifier() -> CardDefinition {
    let text_box = "Whenever another creature you control enters, you gain 1 life.";
    CardDefinition {
        key: "fdn.hinterland_sanctifier",
        scryfall_id: "632df69e-6377-43d0-bba5-65518a320aa5",
        printed: PrintedCharacteristics {
            name: "Hinterland Sanctifier".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rabbit".into(), "Cleric".into()],
            power: Some(1),
            toughness: Some(2),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate::creature()),
                vec![Effect::gain_life(1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sanguine_syphoner() -> CardDefinition {
    let text_box =
        "Whenever this creature attacks, each opponent loses 1 life and you gain 1 life.";
    CardDefinition {
        key: "fdn.sanguine_syphoner",
        scryfall_id: "b1daf5bb-c8e9-4e79-a532-ca92a9a885cd",
        printed: PrintedCharacteristics {
            name: "Sanguine Syphoner".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Warlock".into()],
            power: Some(1),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::DrainLife {
                    amount: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sower_of_chaos() -> CardDefinition {
    let text_box = "{2}{R}: Target creature can't block this turn.";
    CardDefinition {
        key: "fdn.sower_of_chaos",
        scryfall_id: "7ff50606-491c-4946-8d03-719b01cfad77",
        printed: PrintedCharacteristics {
            name: "Sower of Chaos".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Devil".into()],
            power: Some(4),
            toughness: Some(3),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{R}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::CantBlock,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dawnwing_marshal() -> CardDefinition {
    let text_box = "Flying\n{4}{W}: Creatures you control get +1/+1 until end of turn.";
    CardDefinition {
        key: "fdn.dawnwing_marshal",
        scryfall_id: "807db84f-addb-4ef0-b9ba-32b3e1bced3d",
        printed: PrintedCharacteristics {
            name: "Dawnwing Marshal".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}{W}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    }),
                    Modification::pump(1, 1),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn treetop_snarespinner() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)\nDeathtouch (Any amount of damage this deals to a creature is enough to destroy it.)\n{2}{G}: Put a +1/+1 counter on target creature you control. Activate only as a sorcery.";
    CardDefinition {
        key: "fdn.treetop_snarespinner",
        scryfall_id: "88e68fa3-159d-49a6-8ac6-afc9bd6f1718",
        printed: PrintedCharacteristics {
            name: "Treetop Snarespinner".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spider".into()],
            power: Some(1),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Reach, Keyword::Deathtouch]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{G}"),
                sorcery_timing: true,
                effects: vec![Effect::put_counters(
                    ObjectSelector::Target(TargetSpec::CreatureYouControl),
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn apothecary_stomper() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nWhen this creature enters, choose one —\n• Put two +1/+1 counters on target creature you control.\n• You gain 4 life.";
    CardDefinition {
        key: "fdn.apothecary_stomper",
        scryfall_id: "680b7b0c-0e1b-46ce-9917-9fc6e05aa148",
        printed: PrintedCharacteristics {
            name: "Apothecary Stomper".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elephant".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::TriggerModes {
                    up_to: None,
                    unchosen: false,
                    modes: vec![
                        vec![Effect::put_counters(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            CounterKind::PlusOnePlusOne,
                            2,
                        )],
                        vec![Effect::gain_life(4)],
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn spinner_of_souls() -> CardDefinition {
    let text_box = "Reach\nWhenever another nontoken creature you control dies, you may reveal cards from the top of your library until you reveal a creature card. Put that card into your hand and the rest on the bottom of your library in a random order.";
    CardDefinition {
        key: "fdn.spinner_of_souls",
        scryfall_id: "f50a8dec-b079-4192-9098-6cdc1026c693",
        printed: PrintedCharacteristics {
            name: "Spinner of Souls".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spider".into(), "Spirit".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Reach]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::another_you_control(CardPredicate {
                        token: Some(false),
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::RevealUntilToHand {
                            predicate: CardPredicate::creature(),
                        }],
                        vec![],
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn zul_ashur_lich_lord() -> CardDefinition {
    let text_box = "Ward—Pay 2 life. (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays 2 life.)\n{T}: You may cast target Zombie creature card from your graveyard this turn.";
    CardDefinition {
        key: "fdn.zul_ashur_lich_lord",
        scryfall_id: "34ad4fdb-9805-45b3-ba20-e47a15d6ff38",
        printed: PrintedCharacteristics {
            name: "Zul Ashur, Lich Lord".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Zombie".into(), "Warlock".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![StaticAbility::Ward(PaymentCost::Life(2))],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::GrantGraveyardCastThisTurn {
                    flashback: false,
                    target: TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            subtype: Some("Zombie".into()),
                            ..CardPredicate::creature()
                        },
                    },
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn high_fae_trickster() -> CardDefinition {
    let rules_text = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\nYou may cast spells as though they had flash.";
    CardDefinition {
        key: "fdn.high_fae_trickster",
        scryfall_id: "7f1b93ea-1ec1-4010-9343-765742f5088b",
        printed: PrintedCharacteristics {
            name: "High Fae Trickster".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Faerie".into(), "Wizard".into()],
            power: Some(4),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flash, Keyword::Flying]),
            statics: vec![StaticAbility::CastSpellsAsThoughFlash(
                CardPredicate::default(),
            )],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn suspicious_shambler() -> CardDefinition {
    let rules_text = "{4}{B}{B}, Exile this card from your graveyard: Create two 2/2 black Zombie creature tokens. Activate only as a sorcery.";
    CardDefinition {
        key: "fdn.suspicious_shambler",
        scryfall_id: "3d2c5345-eb55-4bca-9183-b4e1404405f8",
        printed: PrintedCharacteristics {
            name: "Suspicious Shambler".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into()],
            power: Some(4),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                zone: ActivationZone::Graveyard,
                mana_cost: ManaCost::parse("{4}{B}{B}"),
                sorcery_timing: true,
                exile_source: true,
                effects: vec![Effect::create_tokens("token.zombie", EffectValue::Fixed(2))],
                ..Default::default()
            }],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn quilled_greatwurm() -> CardDefinition {
    let text_box = "Trample\nWhenever a creature you control deals combat damage during your turn, put that many +1/+1 counters on it. (It must survive to get the counters.)\nYou may cast this card from your graveyard by removing six counters from among creatures you control in addition to paying its other costs.";
    CardDefinition {
        key: "fdn.quilled_greatwurm",
        scryfall_id: "31b60531-3d33-4e66-923a-29008716b15c",
        printed: PrintedCharacteristics {
            name: "Quilled Greatwurm".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Wurm".into()],
            power: Some(7),
            toughness: Some(7),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::YourTurn,
                    condition: Box::new(TriggerCondition::DealsCombatDamage {
                        subject: TriggerSubject::you_control(CardPredicate::creature()),
                        to: None,
                        per: Per::Object,
                    }),
                },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::Triggering,
                    CounterKind::PlusOnePlusOne,
                    EffectValue::TriggeringAmount,
                )],
            }],
            statics: vec![StaticAbility::CastFromGraveyardRemovingCounters(6)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn redcap_gutter_dweller() -> CardDefinition {
    let text_box = "Menace\nWhen this creature enters, create two 1/1 black Rat creature tokens with \"This token can't block.\"\nAt the beginning of your upkeep, you may sacrifice another creature. If you do, put a +1/+1 counter on this creature and exile the top card of your library. You may play that card this turn.";
    CardDefinition {
        key: "fdn.redcap_gutter_dweller",
        scryfall_id: "c6c7f55a-bcc7-4ca3-853d-fc6e260d9429",
        printed: PrintedCharacteristics {
            name: "Redcap Gutter-Dweller".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Menace]),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::create_tokens("token.rat", 2)],
                },
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                    effects: vec![
                        Effect::may_sacrifice(CardPredicate::creature(), true),
                        Effect::if_you_do(vec![
                            Effect::put_counters(
                                ObjectSelector::This,
                                CounterKind::PlusOnePlusOne,
                                1,
                            ),
                            Effect::exile_top_to_play(1, PlayWindow::ThisTurn),
                        ]),
                    ],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn loot_exuberant_explorer() -> CardDefinition {
    let text_box = "You may play an additional land on each of your turns.\n{4}{G}{G}, {T}: Look at the top six cards of your library. You may reveal a creature card with mana value less than or equal to the number of lands you control from among them and put it onto the battlefield. Put the rest on the bottom in a random order.";
    CardDefinition {
        key: "fdn.loot_exuberant_explorer",
        scryfall_id: "09980ce6-425b-4e03-94d0-0f02043cb361",
        printed: PrintedCharacteristics {
            name: "Loot, Exuberant Explorer".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Beast".into(), "Noble".into()],
            power: Some(1),
            toughness: Some(4),
            statics: vec![StaticAbility::AdditionalLandPlay],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}{G}{G}"),
                tap_source: true,
                effects: vec![Effect::LookAndSelect {
                    random_order: true,
                    reveal: true,
                    to_battlefield: true,
                    look: EffectValue::Fixed(6),
                    min_select: 0,
                    max_select: 1,
                    predicate: CardPredicate::creature(),
                    max_mana_value: Some(Box::new(EffectValue::BattlefieldMatching {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Land),
                            ..Default::default()
                        },
                    })),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tinybones_bauble_burglar() -> CardDefinition {
    let text_box = "Whenever an opponent discards a card, exile it from their graveyard with a stash counter on it.\nDuring your turn, you may play cards you don't own with stash counters on them from exile, and mana of any type can be spent to cast those spells.\n{3}{B}, {T}: Each opponent discards a card. Activate only as a sorcery.";
    CardDefinition {
        key: "fdn.tinybones_bauble_burglar",
        scryfall_id: "ff3d85bc-ef2d-4251-baf4-a14bd0cee61e",
        printed: PrintedCharacteristics {
            name: "Tinybones, Bauble Burglar".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Skeleton".into(), "Rogue".into()],
            power: Some(1),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Performs {
                    who: Actor::Opponent,
                    any_of: vec![KeywordAction::Discard],
                },
                effects: vec![Effect::Move {
                    unrevealed: false,
                    library_end: None,
                    what: ObjectSelector::Triggering,
                    to: ZoneType::Exile,
                    tapped: false,
                    transformed: false,
                    under: None,
                    with_counter: Some(CounterKind::Stash),
                    linked: None,
                }],
            }],
            statics: vec![StaticAbility::PlayUnownedExiledWithCounter(
                CounterKind::Stash,
            )],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{B}"),
                tap_source: true,
                sorcery_timing: true,
                effects: vec![Effect::ForPlayer {
                    player: PlayerSelector::EachOpponent,
                    effects: vec![Effect::discard()],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sphinx_of_forgotten_lore() -> CardDefinition {
    let text = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\nWhenever this creature attacks, target instant or sorcery card in your graveyard gains flashback until end of turn. The flashback cost is equal to that card's mana cost. (You may cast that card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "fdn.sphinx_of_forgotten_lore",
        scryfall_id: "af6e46b8-62ed-4bca-ba38-a821f225b59f",
        printed: PrintedCharacteristics {
            name: "Sphinx of Forgotten Lore".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Sphinx".into()],
            power: Some(3),
            toughness: Some(3),
            text_box: text.into(),
            keywords: Keywords::of(&[Keyword::Flash, Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::GrantGraveyardCastThisTurn {
                    flashback: true,
                    target: TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Instant, CardType::Sorcery],
                            ..Default::default()
                        },
                    },
                }],
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}
