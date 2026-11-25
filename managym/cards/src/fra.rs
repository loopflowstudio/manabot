//! Reality Fracture.
use super::enters_prepared;
use managym_state::{
    ability::{
        Ability, Actor, Aggregate, AttackSize, AttackedKind, CompareOp, Condition, DamageKind,
        DamageRecipientFilter, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue,
        Except, KeywordAction, LibraryEnd, ObjectProperty, ObjectSelector, PaymentCost, Per,
        PlayerSelector, Scope, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
        TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, ActivationZone, AdditionalCost, CardDefinition, CardType,
        CardTypes, InsetSpell, Keyword, Keywords, LayerValue, PrintedCharacteristics,
        ReplacementEffect, Restriction, SacrificeCost, StateBasedAction, StaticAbility,
        StaticContinuous, StaticKeywordGrant, StaticPtBuff, StaticRestriction, StaticScope,
    },
    effect::{CopyException, EffectDuration, Modification},
    mana::{Color, Mana, ManaCost, ManaRestriction},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        ajani_resolute(),
        infinite_coursework(),
        paradox_shaper(),
        woodwork_prodigy(),
        stingerquill_voxmancer(),
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
        heartwood_crafter(),
        codie_ravenous_codex(),
        pompous_battlemage(),
        semester_foreseer(),
        theorix_metamage(),
        vigorbloom_vanguard(),
        dedicated_commons(),
        fatehold_annex(),
        formidable_commons(),
        innovative_commons(),
        konstrari_annex(),
        meticulous_commons(),
        stingerquill_annex(),
        theorix_annex(),
        transformative_commons(),
        vigorbloom_annex(),
        void_extrapolator(),
        the_theorist(),
        tomik_orzhov_lawmage(),
        academic_ascent(),
        arcane_amphisbaena(),
        campus_crier(),
        fatehold_charm(),
        hexhaven_battalion(),
        inspired_tethermage(),
        jaces_machinations(),
        keeper_of_the_quiet_hour(),
        mindseeker_oculus(),
        no_admittance(),
        overwrite_the_multiverse(),
        plan_for_all_outcomes(),
        proteges_awakening(),
        repurposed_enforcer(),
        rewrite_regrets(),
        solve_for_disappointment(),
        tams_resistance(),
        theorists_proxy(),
        violent_echoes(),
        vraskas_final_mercy(),
        way_of_the_mentor(),
        way_of_the_necromancer(),
        way_of_the_cryomancer(),
        way_of_the_deathbringer(),
        way_of_the_healer(),
        way_of_the_pyromancer(),
        way_of_the_warlord(),
        way_of_the_wildspeaker(),
        way_of_the_mind_sculptor(),
        way_of_the_paradox(),
        avatar_of_burgeoning_echoes(),
        kiora_of_salt_and_sand(),
        sanctum_lurker(),
        gideon_the_oathless(),
        ajani_unrelenting(),
        garruk_curse_breaker(),
        garruk_veiled_butcher(),
        jace_reality_sculptor(),
        teyo_lightshield_expert(),
        teyo_diamondblade_mage(),
        your_fate_ends_here(),
        terminal_criticism(),
        refute_destiny(),
        extended_absence(),
        fulminous_forte(),
        essence_burn(),
        wrath_of_the_bloodmane(),
        silence_the_echo(),
        break_under_pressure(),
        awaken_the_inferno(),
        craftwork_crusher(),
        compel_brutality(),
        edgar_ancient_bloodlord(),
        liliana_the_faultless(),
        liliana_the_repentant(),
        massacre_girl_most_wanted(),
        winter_tormented_loner(),
        mabel_bitter_recluse(),
        living_library(),
        identity_echo(),
        vindictive_triumph(),
        entrust_the_spark(),
        loyal_tutor(),
        mind_meanderer(),
        lichs_relic(),
        face_yourself(),
    ]
}

fn tomik_orzhov_lawmage() -> CardDefinition {
    let restriction = StaticAbility::Restriction(StaticRestriction {
        scope: StaticScope::This,
        kind: Restriction::CantBeAttackedByMoreThanOne,
        unless: None,
    });
    CardDefinition {
        key: "fra.tomik_orzhov_lawmage",
        scryfall_id: "7ca95235-6e54-4ff8-bc2e-6a3d483ff007",
        printed: PrintedCharacteristics {
            name: "Tomik, Orzhov Lawmage".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Advisor".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: "Flying\nPlaneswalkers you control have \"No more than one creature can attack this planeswalker each combat.\"\n{T}: Target creature with a +1/+1 counter on it gains flying until end of turn.".into(),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::YouControl(CardPredicate {
                    card_type: Some(CardType::Planeswalker),
                    ..Default::default()
                }),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::GrantStatic(Box::new(restriction))],
            })],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Permanent {
                        predicate: CardPredicate { with_counter: true, ..CardPredicate::creature() },
                    }),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
                )],
                ..Default::default()
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn the_theorist() -> CardDefinition {
    let text_box = "At the beginning of each opponent's draw step, you draw a card.\n+1: Create a 1/1 blue Illusion creature token.\n−2: For each opponent, return up to one target artifact or creature that player controls to its owner's hand.\n−6: Draw three cards. Then put X +1/+1 counters on each creature you control, where X is the number of cards in your hand.";
    let bounce = TargetSpec::PermanentOpponentControls {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Artifact, CardType::Creature],
            ..Default::default()
        },
    };
    let loyalty = ActivatedAbilityDefinition::loyalty;
    CardDefinition {
        key: "fra.the_theorist_jace_beleren",
        scryfall_id: "20bb8c55-4b0b-425f-8201-b54fa2fdde86",
        printed: PrintedCharacteristics {
            name: "The Theorist, Jace Beleren".into(),
            types: CardTypes::new([CardType::Planeswalker]),
            subtypes: vec!["Jace".into()],
            supertypes: vec!["Legendary".into()],
            mana_cost: Some(ManaCost::parse("{2}{U}{U}")),
            loyalty: Some(3),
            text_box: text_box.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Draw, Actor::Opponent),
                intervening_if: None,
                effects: vec![Effect::DrawCards { count: 1.into() }],
            }],
            activated_abilities: vec![
                loyalty(1, vec![Effect::create_tokens("token.illusion_1_1_u", 1)]),
                loyalty(
                    -2,
                    vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, bounce.clone()),
                        effects: vec![Effect::return_to_hand(ObjectSelector::Target(bounce))],
                    }],
                ),
                loyalty(
                    -6,
                    vec![
                        Effect::DrawCards { count: 3.into() },
                        Effect::PutCounters {
                            on: ObjectSelector::each_you_control(CardPredicate::creature()),
                            kind: CounterKind::PlusOnePlusOne,
                            amount: EffectValue::CardsInHand,
                        },
                    ],
                ),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn infinite_coursework() -> CardDefinition {
    let text = "Enchant creature\nWhen this Aura enters, tap enchanted creature. It becomes unprepared.\nEnchanted creature loses all abilities and doesn't untap during its controller's untap step.";
    CardDefinition {
        key: "fra.infinite_coursework",
        scryfall_id: "a5988272-faaa-463d-a0a1-a8e96b946bad",
        printed: PrintedCharacteristics {
            name: "Infinite Coursework".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            text_box: text.into(),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::AttachSourceTo {
                what: ObjectSelector::Target(TargetSpec::Creature),
            }],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![
                    Effect::tap(ObjectSelector::Attached),
                    Effect::Unprepare {
                        what: ObjectSelector::Attached,
                    },
                ],
            }],
            statics: vec![
                StaticAbility::Restriction(StaticRestriction {
                    scope: StaticScope::Attached,
                    kind: Restriction::DoesntUntap,
                    unless: None,
                }),
                StaticAbility::Continuous(StaticContinuous {
                    scope: StaticScope::Attached,
                    condition: None,
                    characteristic_defining: false,
                    modifications: vec![Modification::LoseAllAbilities],
                }),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}
fn prepares_at_upkeep(spell: PrintedCharacteristics) -> PrintedCharacteristics {
    PrintedCharacteristics {
        types: CardTypes::new([CardType::Creature]),
        inset: Some(InsetSpell::Prepared(spell.into())),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
            intervening_if: Some(Condition::SourceUnprepared),
            effects: vec![Effect::Prepare {
                what: ObjectSelector::This,
            }],
        }],
        ..Default::default()
    }
}
fn paradox_shaper() -> CardDefinition {
    let text = "At the beginning of your upkeep, if this creature isn't prepared, it becomes prepared.\n{2}: Put target card from your graveyard on the bottom of your library.";
    CardDefinition {
        key: "fra.paradox_shaper",
        scryfall_id: "e61b9d48-0ace-4453-afe0-a1024444bac0",
        printed: PrintedCharacteristics {
            name: "Paradox Shaper".into(),
            mana_cost: Some(ManaCost::parse("{1}{U/B}")),
            subtypes: vec!["Octopus".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(3),
            text_box: text.into(),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::put_in_library(
                    ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate::default(),
                    }),
                    LibraryEnd::Bottom,
                )],
                ..Default::default()
            }],
            ..prepares_at_upkeep(omit_variables())
        },
        ..Default::default()
    }
}
fn stingerquill_voxmancer() -> CardDefinition {
    let text = "At the beginning of your upkeep, if this creature isn't prepared, it becomes prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "fra.stingerquill_voxmancer",
        scryfall_id: "84b1c268-3b8a-41b6-92e3-a2ce0cc3d738",
        printed: PrintedCharacteristics {
            name: "Stingerquill Voxmancer".into(),
            mana_cost: Some(ManaCost::parse("{B/R}")),
            subtypes: vec!["Goblin".into(), "Sorcerer".into()],
            power: Some(1),
            toughness: Some(2),
            text_box: text.into(),

            ..prepares_at_upkeep(vicious_verse())
        },
        ..Default::default()
    }
}
fn woodwork_prodigy() -> CardDefinition {
    let text = "At the beginning of your upkeep, if this creature isn't prepared, it becomes prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "fra.woodwork_prodigy",
        scryfall_id: "7d17f7e3-7b63-4674-9024-4fd1827f40ec",
        printed: PrintedCharacteristics {
            name: "Woodwork Prodigy".into(),
            mana_cost: Some(ManaCost::parse("{2}{R/G}")),
            subtypes: vec!["Cat".into(), "Druid".into()],
            power: Some(3),
            toughness: Some(3),
            text_box: text.into(),

            ..prepares_at_upkeep(soul_tether())
        },
        ..Default::default()
    }
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
                    to: Some(Actor::AnyPlayer),
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

fn heartwood_crafter() -> CardDefinition {
    let text_box = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\n{T}: Add {C}. This mana can't be spent to cast spells from your hand.";
    CardDefinition {
        key: "fra.heartwood_crafter",
        scryfall_id: "910a1f41-17fd-4ab0-9597-7151e79dc760",
        printed: PrintedCharacteristics {
            name: "Heartwood Crafter".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            text_box: text_box.into(),
            subtypes: vec!["Elf".into(), "Artificer".into()],
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::restricted_to(
                    ManaRestriction::ExceptSpellsFromHand,
                    [0, 0, 0, 0, 0, 1],
                ))],
                ..Default::default()
            }],
            ..enters_prepared(soul_tether())
        },
        ..Default::default()
    }
}

fn codie_ravenous_codex() -> CardDefinition {
    let text_box = "Whenever you cast a prepared spell, copy it. You may choose new targets for the copy.\n{W}{U}{B}{R}{G}, {T}: Each creature you control becomes prepared. (Only creatures with prepare spells can become prepared.)";
    CardDefinition {
        key: "fra.codie_ravenous_codex",
        scryfall_id: "c3192390-1518-49fc-8716-f2c7a0384f39",
        printed: PrintedCharacteristics {
            name: "Codie, Ravenous Codex".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Book".into(), "Construct".into()],
            power: Some(1),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Prepared {
                    condition: Box::new(TriggerCondition::casts(
                        Actor::You,
                        CardPredicate::default(),
                    )),
                },
                intervening_if: None,
                effects: vec![Effect::CopyTriggeringObject],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{W}{U}{B}{R}{G}"),
                tap_source: true,
                effects: vec![Effect::Prepare {
                    what: ObjectSelector::each_you_control(CardPredicate::creature()),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
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
            player: None,
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

fn planeswalker_land(
    key: &'static str,
    scryfall_id: &'static str,
    name: &str,
    colors: [&str; 2],
) -> CardDefinition {
    let [first, second] = colors;
    CardDefinition {
        key,
        scryfall_id,
        printed: PrintedCharacteristics {
            name: name.into(),
            types: CardTypes::new([CardType::Land]),
            replacement_effects: vec![ReplacementEffect::EntersTappedUnlessControls {
                predicate: CardPredicate {
                    card_type: Some(CardType::Planeswalker),
                    ..Default::default()
                },
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(colors.map(Mana::parse))],
                ..Default::default()
            }],
            text_box: format!("This land enters tapped unless you control a planeswalker.\n{{T}}: Add {{{first}}} or {{{second}}}."),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dedicated_commons() -> CardDefinition {
    planeswalker_land(
        "fra.dedicated_commons",
        "3223e5db-5cc4-42f9-ae9e-ff58abc7c390",
        "Dedicated Commons",
        ["R", "W"],
    )
}

fn fatehold_annex() -> CardDefinition {
    planeswalker_land(
        "fra.fatehold_annex",
        "5140f962-62f3-40fd-a322-44896c7e2613",
        "Fatehold Annex",
        ["W", "U"],
    )
}

fn formidable_commons() -> CardDefinition {
    planeswalker_land(
        "fra.formidable_commons",
        "e6ca6c3e-f145-42d6-8a17-90770c15afaf",
        "Formidable Commons",
        ["B", "G"],
    )
}

fn innovative_commons() -> CardDefinition {
    planeswalker_land(
        "fra.innovative_commons",
        "84ea799a-faa2-4ff1-a933-432d4ee31a3b",
        "Innovative Commons",
        ["U", "R"],
    )
}

fn konstrari_annex() -> CardDefinition {
    planeswalker_land(
        "fra.konstrari_annex",
        "39c805e3-82cd-42a9-80fe-8d81712a94ea",
        "Konstrari Annex",
        ["R", "G"],
    )
}

fn meticulous_commons() -> CardDefinition {
    planeswalker_land(
        "fra.meticulous_commons",
        "93ac525e-1919-43dd-aba4-073b7e4c1768",
        "Meticulous Commons",
        ["W", "B"],
    )
}

fn stingerquill_annex() -> CardDefinition {
    planeswalker_land(
        "fra.stingerquill_annex",
        "6ede3143-69ac-4cbe-922a-d25b07c26da7",
        "Stingerquill Annex",
        ["B", "R"],
    )
}

fn theorix_annex() -> CardDefinition {
    planeswalker_land(
        "fra.theorix_annex",
        "97bbbd23-ecb1-4407-ac14-dede08532a1e",
        "Theorix Annex",
        ["U", "B"],
    )
}

fn transformative_commons() -> CardDefinition {
    planeswalker_land(
        "fra.transformative_commons",
        "b57d5be7-3157-4b49-aeb8-d7368ca7e9dd",
        "Transformative Commons",
        ["G", "U"],
    )
}

fn vigorbloom_annex() -> CardDefinition {
    planeswalker_land(
        "fra.vigorbloom_annex",
        "db8c7bdd-76cd-4be0-ae0d-d430e9a5fe7a",
        "Vigorbloom Annex",
        ["G", "W"],
    )
}

fn academic_ascent() -> CardDefinition {
    let text_box = "Target creature gets +2/+2 and gains flying until end of turn.\nEmpower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let mut effects = Effect::pump_and_grant_until_end_of_turn(
        ObjectSelector::Target(TargetSpec::Creature),
        2,
        2,
        Keyword::Flying,
    );
    effects.extend(Effect::empower_jace(2));
    CardDefinition {
        key: "fra.academic_ascent",
        scryfall_id: "730d8c28-1e58-4b8e-89e9-445d154d2e83",
        printed: PrintedCharacteristics {
            name: "Academic Ascent".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn arcane_amphisbaena() -> CardDefinition {
    let text_box = "Deathtouch\nWhen this creature enters, empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.arcane_amphisbaena",
        scryfall_id: "1bf923c4-f0b7-4271-978c-fd2e79fe1cc8",
        printed: PrintedCharacteristics {
            name: "Arcane Amphisbaena".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Snake".into()],
            power: Some(1),
            toughness: Some(1),
            text_box: text_box.into(),
            abilities: vec![Ability::enters(Effect::empower_jace(2))],
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn campus_crier() -> CardDefinition {
    let text_box = "{1}, Exile this card from your graveyard: Empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.campus_crier",
        scryfall_id: "6047b14c-91d5-4f8e-af3f-057a541e2546",
        printed: PrintedCharacteristics {
            name: "Campus Crier".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Advisor".into()],
            power: Some(3),
            toughness: Some(1),
            text_box: text_box.into(),
            activated_abilities: vec![ActivatedAbilityDefinition {
                zone: ActivationZone::Graveyard,
                exile_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effects: Effect::empower_jace(2),
                ..Default::default()
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fatehold_charm() -> CardDefinition {
    let text_box = "Choose one —\n• Draw a card. Empower Jace 2.\n• Return target spell or creature to its owner's hand.\n• Creatures you control get +1/+2 until end of turn.";
    let mut draw = vec![Effect::DrawCards { count: 1.into() }];
    draw.extend(Effect::empower_jace(2));
    CardDefinition {
        key: "fra.fatehold_charm",
        scryfall_id: "cfc54011-647e-4428-bcdb-59400e1da49d",
        printed: PrintedCharacteristics {
            name: "Fatehold Charm".into(),
            mana_cost: Some(ManaCost::parse("{W}{U}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: text_box.into(),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    draw,
                    vec![Effect::return_to_hand(ObjectSelector::Target(
                        TargetSpec::CreatureOrSpell,
                    ))],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each_you_control(CardPredicate::creature()),
                        Modification::pump(1, 2),
                    )],
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hexhaven_battalion() -> CardDefinition {
    let text_box = "Create three 2/2 colorless Wizard Soldier creature tokens named Cadet. Empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\nBasic landcycling {2} ({2}, Discard this card: Search your library for a basic land card, reveal it, put it into your hand, then shuffle.)";
    let mut effects = vec![Effect::create_tokens("token.cadet", 3)];
    effects.extend(Effect::empower_jace(2));
    CardDefinition {
        key: "fra.hexhaven_battalion",
        scryfall_id: "3b6ac80e-c726-4bd0-893a-e666041a04a6",
        printed: PrintedCharacteristics {
            name: "Hexhaven Battalion".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            activated_abilities: vec![ActivatedAbilityDefinition {
                zone: ActivationZone::Hand,
                discard_source: true,
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::search_library(
                    CardPredicate::basic_land(),
                    1.into(),
                    vec![Effect::return_to_hand(ObjectSelector::Selected)],
                )],
                ..Default::default()
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn inspired_tethermage() -> CardDefinition {
    let text_box = "Whenever you put one or more loyalty counters on a planeswalker, put a +1/+1 counter on this creature.\n{6}: Empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.inspired_tethermage",
        scryfall_id: "073f4998-a204-447b-93d5-746ae87fd6a1",
        printed: PrintedCharacteristics {
            name: "Inspired Tethermage".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(2),
            text_box: text_box.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::PutsCounters {
                    who: Actor::You,
                    kind: CounterKind::Loyalty,
                    on: CardPredicate {
                        card_type: Some(CardType::Planeswalker),
                        ..Default::default()
                    },
                },
                intervening_if: None,
                effects: vec![Effect::PutCounters {
                    on: ObjectSelector::This,
                    kind: CounterKind::PlusOnePlusOne,
                    amount: 1.into(),
                }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{6}"),
                effects: Effect::empower_jace(2),
                ..Default::default()
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn jaces_machinations() -> CardDefinition {
    let text_box = "Until end of turn, you may activate loyalty abilities of Jace planeswalkers you control on any player's turn any time you could cast an instant.\nEmpower Jace 8. (Put eight loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let mut effects = vec![Effect::until_end_of_turn(
        ObjectSelector::You,
        Modification::LoyaltyAtInstantSpeed(Box::new(CardPredicate {
            card_type: Some(CardType::Planeswalker),
            subtype: Some("Jace".into()),
            ..Default::default()
        })),
    )];
    effects.extend(Effect::empower_jace(8));
    CardDefinition {
        key: "fra.jaces_machinations",
        scryfall_id: "282588b9-3656-453b-aa25-2419e078ddc1",
        printed: PrintedCharacteristics {
            name: "Jace's Machinations".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn keeper_of_the_quiet_hour() -> CardDefinition {
    let text_box = "When this creature enters, empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.keeper_of_the_quiet_hour",
        scryfall_id: "b6331218-dbb8-44a9-8ba5-5fb1ab41c5c0",
        printed: PrintedCharacteristics {
            name: "Keeper of the Quiet Hour".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Chimera".into()],
            power: Some(3),
            toughness: Some(2),
            text_box: text_box.into(),
            abilities: vec![Ability::enters(Effect::empower_jace(2))],

            ..Default::default()
        },
        ..Default::default()
    }
}

fn mindseeker_oculus() -> CardDefinition {
    let text_box = "When this creature enters, empower Jace 4. (Put four loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.mindseeker_oculus",
        scryfall_id: "f5324741-353a-4a70-adb2-b631b00806dd",
        printed: PrintedCharacteristics {
            name: "Mindseeker Oculus".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Homunculus".into()],
            power: Some(2),
            toughness: Some(1),
            text_box: text_box.into(),
            abilities: vec![Ability::enters(Effect::empower_jace(4))],

            ..Default::default()
        },
        ..Default::default()
    }
}

fn no_admittance() -> CardDefinition {
    let text_box = "No Admittance deals 3 damage to any target.\nEmpower Jace 1. (Put a loyalty counter on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let mut effects = vec![Effect::deal_damage(
        3,
        ObjectSelector::Target(TargetSpec::AnyTarget),
    )];
    effects.extend(Effect::empower_jace(1));
    CardDefinition {
        key: "fra.no_admittance",
        scryfall_id: "11ba4fdd-cc03-4bb6-a493-91a9785771d0",
        printed: PrintedCharacteristics {
            name: "No Admittance".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn overwrite_the_multiverse() -> CardDefinition {
    let text_box = "Exile all creatures. Empower Jace X, where X is the number of creatures exiled this way. (Put that many loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let mut effects = vec![Effect::exile(ObjectSelector::each(
        Scope::Battlefield,
        CardPredicate::creature(),
    ))];
    effects.extend(Effect::empower_jace(EffectValue::Objects {
        of: Box::new(ObjectSelector::Moved),
        predicate: None,
        in_zone: Some(ZoneType::Exile),
    }));
    CardDefinition {
        key: "fra.overwrite_the_multiverse",
        scryfall_id: "c4554f5b-791b-48f6-bf54-ad28699e1beb",
        printed: PrintedCharacteristics {
            name: "Overwrite the Multiverse".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn plan_for_all_outcomes() -> CardDefinition {
    let text_box = "When this enchantment enters, the owner of up to one other target nonland permanent puts it on their choice of the top or bottom of their library.\nWhenever you cast your first noncreature spell each turn, empower Jace 1. (Put a loyalty counter on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let spec = TargetSpec::Permanent {
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "fra.plan_for_all_outcomes",
        scryfall_id: "c4effc17-0d0e-423a-b5f2-597ea6c71f67",
        printed: PrintedCharacteristics {
            name: "Plan for All Outcomes".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            text_box: text_box.into(),
            abilities: vec![
                Ability::enters(vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        another: true,
                        ..TargetRequirement::up_to(1, spec.clone())
                    },
                    effects: vec![Effect::OwnerPutsIntoLibrary {
                        target: spec,
                        top: true,
                    }],
                }]),
                Ability::Triggered {
                    condition: TriggerCondition::Casts {
                        who: Actor::You,
                        predicate: CardPredicate::noncreature(),
                        nth: Some(1),
                        targets: managym_state::ability::SpellTargetFilter::Any,
                    },
                    intervening_if: None,
                    effects: Effect::empower_jace(1),
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn proteges_awakening() -> CardDefinition {
    let text_box = "Empower Jace 6. (Put six loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\nDraw a card.";
    let mut effects = Effect::empower_jace(6);
    effects.push(Effect::DrawCards { count: 1.into() });
    CardDefinition {
        key: "fra.proteges_awakening",
        scryfall_id: "a08c7ec2-4c6a-4db2-85a7-41afe8731523",
        printed: PrintedCharacteristics {
            name: "Protege's Awakening".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn repurposed_enforcer() -> CardDefinition {
    let text_box = "Whenever this creature attacks, empower Jace X, where X is the number of creatures you control. (Put that many loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.repurposed_enforcer",
        scryfall_id: "35000e93-85d3-44f8-976a-5918ee4c71e0",
        printed: PrintedCharacteristics {
            name: "Repurposed Enforcer".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(3),
            toughness: Some(2),
            text_box: text_box.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: Effect::empower_jace(EffectValue::BattlefieldMatching {
                    predicate: CardPredicate::creature(),
                }),
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rewrite_regrets() -> CardDefinition {
    let text_box = "Return target creature or planeswalker card with mana value 6 or less from your graveyard to the battlefield.\nEmpower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let mut effects = vec![Effect::return_to_battlefield(
        ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                max_mana_value: Some(6),
                ..Default::default()
            },
        }),
        false,
    )];
    effects.extend(Effect::empower_jace(2));
    CardDefinition {
        key: "fra.rewrite_regrets",
        scryfall_id: "453cfde7-c460-4b55-9472-b714e16f24bb",
        printed: PrintedCharacteristics {
            name: "Rewrite Regrets".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn solve_for_disappointment() -> CardDefinition {
    let text_box = "Target opponent reveals their hand. You choose a nonland permanent card from it. That player discards that card.\nEmpower Jace 1. (Put a loyalty counter on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let mut effects = vec![Effect::RevealHandDiscardChosen {
        target: TargetSpec::Opponent,
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land, CardType::Instant, CardType::Sorcery],
            ..Default::default()
        },
    }];
    effects.extend(Effect::empower_jace(1));
    CardDefinition {
        key: "fra.solve_for_disappointment",
        scryfall_id: "7beaa8c9-1a2c-4c88-b579-91e371d8d9e3",
        printed: PrintedCharacteristics {
            name: "Solve for Disappointment".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tams_resistance() -> CardDefinition {
    let text_box = "Put a +1/+1 counter on up to one target creature. It gains vigilance until end of turn.\nEmpower Jace 4. (Put four loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let creature = ObjectSelector::Target(TargetSpec::Creature);
    let mut effects = vec![Effect::TargetClause {
        requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
        effects: vec![
            Effect::PutCounters {
                on: creature.clone(),
                kind: CounterKind::PlusOnePlusOne,
                amount: 1.into(),
            },
            Effect::until_end_of_turn(
                creature,
                Modification::AddKeywords(Keywords::of(&[Keyword::Vigilance])),
            ),
        ],
    }];
    effects.extend(Effect::empower_jace(4));
    CardDefinition {
        key: "fra.tams_resistance",
        scryfall_id: "b3d33df2-a77b-4c2e-ba7f-2cc9c1505f9b",
        printed: PrintedCharacteristics {
            name: "Tam's Resistance".into(),
            mana_cost: Some(ManaCost::parse("{1}{G/U}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: effects,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn theorists_proxy() -> CardDefinition {
    let text_box = "Flash\nWhen this creature enters, empower Jace 3. (Put three loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\n{U}, Sacrifice this creature: The next spell you cast this turn can't be countered.";
    CardDefinition {
        key: "fra.theorists_proxy",
        scryfall_id: "710302ca-c4be-4069-8ce1-f531414c74e9",
        printed: PrintedCharacteristics {
            name: "Theorist's Proxy".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Illusion".into()],
            power: Some(0),
            toughness: Some(3),
            text_box: text_box.into(),
            abilities: vec![Ability::enters(Effect::empower_jace(3))],
            keywords: Keywords::of(&[Keyword::Flash]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{U}"),
                sacrifice_source: true,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::NextSpellYouCast,
                    Modification::CantBeCountered,
                )],
                ..Default::default()
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn violent_echoes() -> CardDefinition {
    let text_box = "Violent Echoes deals 6 damage to target creature or planeswalker. If excess damage was dealt to that permanent this way, empower Jace X, where X is that excess damage. (Put that many loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    CardDefinition {
        key: "fra.violent_echoes",
        scryfall_id: "ad03ba90-2442-4a71-94df-2088b5b63662",
        printed: PrintedCharacteristics {
            name: "Violent Echoes".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: text_box.into(),
            spell_effects: vec![
                Effect::deal_damage(
                    6,
                    ObjectSelector::Target(TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                            ..Default::default()
                        },
                    }),
                ),
                Effect::If {
                    condition: Condition::compare(EffectValue::ExcessDamage, CompareOp::Gt, 0),
                    then: Effect::empower_jace(EffectValue::ExcessDamage),
                    otherwise: vec![],
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vraskas_final_mercy() -> CardDefinition {
    let text_box = "Choose one —\n• You lose 2 life. Destroy target creature or planeswalker.\n• You lose 2 life. Empower Jace 6. (Put six loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")";
    let lose_life = Effect::LoseLife {
        player: PlayerSelector::You,
        amount: 2.into(),
    };
    let mut empower = vec![lose_life.clone()];
    empower.extend(Effect::empower_jace(6));
    CardDefinition {
        key: "fra.vraskas_final_mercy",
        scryfall_id: "992bd991-7cfb-459f-bafd-9a44f3c925c5",
        printed: PrintedCharacteristics {
            name: "Vraska's Final Mercy".into(),
            mana_cost: Some(ManaCost::parse("{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![
                        lose_life,
                        Effect::Destroy {
                            what: ObjectSelector::Target(TargetSpec::Permanent {
                                predicate: CardPredicate {
                                    card_types_any: vec![
                                        CardType::Creature,
                                        CardType::Planeswalker,
                                    ],
                                    ..Default::default()
                                },
                            }),
                        },
                    ],
                    empower,
                ],
            }],
            ..Default::default()
        },
        ..Default::default()
    }
}

/// "Planeswalkers you control" (CR 306), the scope every Way grants to.
fn planeswalker() -> CardPredicate {
    CardPredicate::card_type(CardType::Planeswalker)
}

/// "Planeswalkers you control have \"[ability]\"": a layer-6 grant (CR
/// 613.1f) each holder activates as its own loyalty ability, sharing that
/// holder's one loyalty activation each turn (CR 606.3).
fn planeswalkers_you_control_have(ability: ActivatedAbilityDefinition) -> StaticAbility {
    StaticAbility::continuous(
        StaticScope::YouControl(planeswalker()),
        None,
        vec![Modification::GrantActivated(Box::new(ability))],
    )
}

/// The reminder text of "empower Jace N" in the Way cycle's first ability.
fn empower_reminder(name: &str, count: &str) -> String {
    format!(
        "When {name} enters, empower Jace {count_digit}. (Put {count} loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")",
        count_digit = match count {
            "two" => "2",
            "five" => "5",
            "seven" => "7",
            _ => unreachable!("no Way empowers {count}"),
        },
    )
}

/// A Way: a legendary enchantment that empowers Jace as it enters, then
/// either grants planeswalkers you control a loyalty ability or watches them.
struct Way {
    key: &'static str,
    scryfall_id: &'static str,
    name: &'static str,
    mana_cost: &'static str,
    empower: &'static str,
    text: &'static str,
    abilities: Vec<Ability>,
    statics: Vec<StaticAbility>,
}

fn way(way: Way) -> CardDefinition {
    let amount = match way.empower {
        "two" => 2,
        "five" => 5,
        "seven" => 7,
        _ => unreachable!("no Way empowers {}", way.empower),
    };
    let text_box = format!("{}\n{}", empower_reminder(way.name, way.empower), way.text);
    let mut abilities = vec![Ability::enters(Effect::empower_jace(amount))];
    abilities.extend(way.abilities);
    CardDefinition {
        key: way.key,
        scryfall_id: way.scryfall_id,
        printed: PrintedCharacteristics {
            name: way.name.into(),
            mana_cost: Some(ManaCost::parse(way.mana_cost)),
            types: CardTypes::new([CardType::Enchantment]),
            supertypes: vec!["Legendary".into()],
            text_box: text_box.into(),
            abilities,
            statics: way.statics,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn way_of_the_mentor() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_mentor",
        scryfall_id: "1a59d5b1-12d6-486b-bd29-ca371359addd",
        name: "Way of the Mentor",
        mana_cost: "{2}{W}",
        empower: "five",
        text: "Whenever you gain life, put a loyalty counter on each planeswalker you control.",
        abilities: vec![Ability::when_you_gain_life(Effect::put_counters(
            ObjectSelector::each_you_control(planeswalker()),
            CounterKind::Loyalty,
            1,
        ))],
        statics: Vec::new(),
    })
}

fn way_of_the_necromancer() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_necromancer",
        scryfall_id: "a0ff9689-ea49-4fff-b37c-4abbaeb0f73d",
        name: "Way of the Necromancer",
        mana_cost: "{1}{B}",
        empower: "two",
        text: "Whenever a creature you control dies, put a loyalty counter on each planeswalker you control.",
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::you_control(CardPredicate::creature()),
            },
            intervening_if: None,
            effects: vec![Effect::put_counters(
                ObjectSelector::each_you_control(planeswalker()),
                CounterKind::Loyalty,
                1,
            )],
        }],
        statics: Vec::new(),
    })
}

fn way_of_the_cryomancer() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_cryomancer",
        scryfall_id: "838b0efb-7398-4df9-8fdf-b8af43b47938",
        name: "Way of the Cryomancer",
        mana_cost: "{2}{U}",
        empower: "five",
        text: "Planeswalkers you control have \"[−3]: When you next cast an instant or sorcery spell this turn, copy that spell. You may choose new targets for the copy.\"",
        abilities: Vec::new(),
        statics: vec![planeswalkers_you_control_have(
            ActivatedAbilityDefinition::loyalty(
                -3,
                vec![Effect::CreateDelayedTrigger {
                    remember: None,
                    event: DelayedEvent::YouCastSpell(Box::new(CardPredicate::instant_or_sorcery())),
                    frequency: DelayedFrequency::Once,
                    expiry: DelayedExpiry::ThisTurn,
                    effects: vec![Effect::CopyTriggeringObject],
                }],
            ),
        )],
    })
}

fn way_of_the_deathbringer() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_deathbringer",
        scryfall_id: "12dd46b2-e892-4660-b120-55766fd4d878",
        name: "Way of the Deathbringer",
        mana_cost: "{2}{B}",
        empower: "five",
        text: "Planeswalkers you control have \"[−2]: You may sacrifice a creature. If you do, create a 4/4 green Beast creature token with trample.\"",
        abilities: Vec::new(),
        statics: vec![planeswalkers_you_control_have(
            ActivatedAbilityDefinition::loyalty(
                -2,
                vec![
                    Effect::may_sacrifice(CardPredicate::creature(), false),
                    Effect::if_you_do(vec![Effect::create_tokens("token.beast_4_4_g_trample", 1)]),
                ],
            ),
        )],
    })
}

fn way_of_the_healer() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_healer",
        scryfall_id: "50326a2a-7e10-464b-a97e-e880bda0558c",
        name: "Way of the Healer",
        mana_cost: "{3}{W}",
        empower: "five",
        text: "Planeswalkers you control have \"[−2]: Create a 2/2 colorless Wizard Soldier creature token named Cadet. Surveil 1.\"",
        abilities: Vec::new(),
        statics: vec![planeswalkers_you_control_have(
            ActivatedAbilityDefinition::loyalty(
                -2,
                vec![
                    Effect::create_tokens("token.cadet", 1),
                    Effect::Surveil { count: 1 },
                ],
            ),
        )],
    })
}

fn way_of_the_pyromancer() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_pyromancer",
        scryfall_id: "c1a00020-7c14-4503-a057-5763704bb83e",
        name: "Way of the Pyromancer",
        mana_cost: "{1}{R}",
        empower: "two",
        text: "Planeswalkers you control have \"[+1]: Add {R}.\"",
        abilities: Vec::new(),
        // A loyalty ability is never a mana ability (CR 605.1a): it uses the stack.
        statics: vec![planeswalkers_you_control_have(
            ActivatedAbilityDefinition::loyalty(1, vec![Effect::add_mana(Mana::parse("R"))]),
        )],
    })
}

fn way_of_the_warlord() -> CardDefinition {
    let creature_or_planeswalker = TargetSpec::Permanent {
        predicate: CardPredicate::creature_or_planeswalker(),
    };
    way(Way {
        key: "fra.way_of_the_warlord",
        scryfall_id: "6d86e410-20c4-4248-96bf-5780ece6274a",
        name: "Way of the Warlord",
        mana_cost: "{2}{R}",
        empower: "five",
        text: "Planeswalkers you control have \"[−4]: This planeswalker deals 2 damage to up to one target creature or planeswalker and 2 damage to target player.\"",
        abilities: Vec::new(),
        // Two independent target clauses (CR 601.2c): the player is still
        // dealt damage when no creature or planeswalker is chosen.
        statics: vec![planeswalkers_you_control_have(
            ActivatedAbilityDefinition::loyalty(
                -4,
                vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, creature_or_planeswalker.clone()),
                        effects: vec![Effect::deal_damage(
                            2,
                            ObjectSelector::Target(creature_or_planeswalker),
                        )],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(TargetSpec::Player),
                        effects: vec![Effect::DealDamageToPlayer {
                            player: PlayerSelector::Target(Box::new(TargetSpec::Player)),
                            amount: 2.into(),
                        }],
                    },
                ],
            ),
        )],
    })
}

fn way_of_the_wildspeaker() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_wildspeaker",
        scryfall_id: "a252cb01-537b-4afe-9abc-81a98c4a1439",
        name: "Way of the Wildspeaker",
        mana_cost: "{4}{G}",
        empower: "seven",
        text: "Planeswalkers you control have \"[−4]: Create a 4/4 green Beast creature token with trample.\"",
        abilities: Vec::new(),
        statics: vec![planeswalkers_you_control_have(
            ActivatedAbilityDefinition::loyalty(
                -4,
                vec![Effect::create_tokens("token.beast_4_4_g_trample", 1)],
            ),
        )],
    })
}

fn way_of_the_mind_sculptor() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_mind_sculptor",
        scryfall_id: "5838af68-66c3-4fe8-ab89-0a1721b0cfeb",
        name: "Way of the Mind Sculptor",
        mana_cost: "{4}{U}",
        empower: "five",
        text: "Whenever you activate a loyalty ability, if you removed two or more loyalty counters to activate it, draw a card.",
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActivatesLoyaltyAbility {
                who: Actor::You,
                min_counters_removed: 2,
            },
            intervening_if: None,
            effects: vec![Effect::DrawCards { count: 1.into() }],
        }],
        statics: Vec::new(),
    })
}

fn way_of_the_paradox() -> CardDefinition {
    way(Way {
        key: "fra.way_of_the_paradox",
        scryfall_id: "98dc5470-507a-4364-8480-42607255e56c",
        name: "Way of the Paradox",
        mana_cost: "{2}{G}",
        empower: "five",
        text: "Whenever you activate a loyalty ability, you gain 1 life. You may play an additional land this turn.",
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActivatesLoyaltyAbility {
                who: Actor::You,
                min_counters_removed: 0,
            },
            intervening_if: None,
            effects: vec![
                Effect::gain_life(1),
                Effect::until_end_of_turn(ObjectSelector::You, Modification::AdditionalLandPlay),
            ],
        }],
        statics: Vec::new(),
    })
}

fn avatar_of_burgeoning_echoes() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\nPlaneswalkers you control have \"[−10]: Put a +1/+1 counter on target creature for each land you control.\"";
    let land = CardPredicate::card_type(CardType::Land);
    CardDefinition {
        key: "fra.avatar_of_burgeoning_echoes",
        scryfall_id: "5905995b-7a20-4602-a7cc-90aa5089a082",
        printed: PrintedCharacteristics {
            name: "Avatar of Burgeoning Echoes".into(),
            mana_cost: Some(ManaCost::parse("{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Avatar".into()],
            power: Some(2),
            toughness: Some(3),
            text_box: text_box.into(),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(land.clone()),
                Effect::empower_jace(2),
            )],
            statics: vec![planeswalkers_you_control_have(
                ActivatedAbilityDefinition::loyalty(
                    -10,
                    vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::Creature),
                        CounterKind::PlusOnePlusOne,
                        ObjectSelector::each_you_control(land).count(),
                    )],
                ),
            )],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kiora_of_salt_and_sand() -> CardDefinition {
    let text_box = "Whenever you attack, if you've activated a loyalty ability this turn, untap target attacking creature. It can't be blocked this turn.\nPlaneswalkers you control have \"[−8]: Create an 8/8 blue Leviathan creature token with hexproof.\"";
    let attacking = TargetSpec::Permanent {
        predicate: CardPredicate {
            attacking: true,
            ..CardPredicate::creature()
        },
    };
    CardDefinition {
        key: "fra.kiora_of_salt_and_sand",
        scryfall_id: "8151f5f5-e9f6-4fbe-b543-f456ebf22aa5",
        printed: PrintedCharacteristics {
            name: "Kiora of Salt and Sand".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Merfolk".into(), "Noble".into()],
            power: Some(2),
            toughness: Some(4),
            text_box: text_box.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Batch,
                ),
                intervening_if: Some(Condition::compare(
                    EffectValue::LoyaltyActivationsThisTurn(PlayerSelector::You),
                    CompareOp::Ge,
                    1,
                )),
                effects: vec![
                    Effect::untap(ObjectSelector::Target(attacking.clone())),
                    Effect::until_end_of_turn(
                        ObjectSelector::Target(attacking),
                        Modification::CantBeBlocked,
                    ),
                ],
            }],
            statics: vec![planeswalkers_you_control_have(
                ActivatedAbilityDefinition::loyalty(
                    -8,
                    vec![Effect::create_tokens("token.leviathan_8_8_u_hexproof", 1)],
                ),
            )],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sanctum_lurker() -> CardDefinition {
    let text_box = "When this creature enters, empower Jace 1. (Put a loyalty counter on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\nPlaneswalkers you control aren't put into their owners' graveyards for having 0 loyalty.\nPlaneswalkers you control have \"[+2]: This planeswalker deals 1 damage to each opponent and you gain 1 life.\"";
    CardDefinition {
        key: "fra.sanctum_lurker",
        scryfall_id: "2185c08f-bb4d-49d5-8b6c-c629a48bb61c",
        printed: PrintedCharacteristics {
            name: "Sanctum Lurker".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Horror".into()],
            power: Some(3),
            toughness: Some(2),
            text_box: text_box.into(),
            abilities: vec![Ability::enters(Effect::empower_jace(1))],
            statics: vec![
                StaticAbility::StateBasedActionDoesntApply {
                    action: StateBasedAction::ZeroLoyalty,
                    scope: StaticScope::YouControl(planeswalker()),
                },
                planeswalkers_you_control_have(ActivatedAbilityDefinition::loyalty(
                    2,
                    vec![
                        Effect::DealDamageToPlayer {
                            player: PlayerSelector::EachOpponent,
                            amount: 1.into(),
                        },
                        Effect::gain_life(1),
                    ],
                )),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gideon_the_oathless() -> CardDefinition {
    let text_box = "Ward—Discard a card.\nWhenever a creature an opponent controls enters, Gideon deals 1 damage to that player.\nWhenever an opponent activates a loyalty ability, Gideon deals 1 damage to that player.";
    let damage_that_player = || Effect::DealDamageToPlayer {
        player: PlayerSelector::TriggeringPlayer,
        amount: 1.into(),
    };
    CardDefinition {
        key: "fra.gideon_the_oathless",
        scryfall_id: "c985b0d1-25bd-4069-aab7-a566ff27a8f6",
        printed: PrintedCharacteristics {
            name: "Gideon the Oathless".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Mercenary".into()],
            power: Some(3),
            toughness: Some(3),
            text_box: text_box.into(),
            statics: vec![StaticAbility::Ward(PaymentCost::Discard)],
            abilities: vec![
                Ability::when_enters(
                    TriggerSubject::matching(Actor::Opponent, CardPredicate::creature(), false),
                    vec![damage_that_player()],
                ),
                Ability::Triggered {
                    condition: TriggerCondition::ActivatesLoyaltyAbility {
                        who: Actor::Opponent,
                        min_counters_removed: 0,
                    },
                    intervening_if: None,
                    effects: vec![damage_that_player()],
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ajani_resolute() -> CardDefinition {
    let text_box = "Whenever you gain life, put a loyalty counter on Ajani.\n0: You gain 1 life.\n−4: Create a 2/2 white Cat Soldier creature token named Ajani's Pridemate with \"Whenever you gain life, put a +1/+1 counter on this token.\"\n−10: You get an emblem with \"Creatures you control get +2/+2.\"";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    CardDefinition {
        key: "fra.ajani_resolute",
        scryfall_id: "a5e1a7dd-8c49-4435-935c-bcc78704082b",
        printed: PrintedCharacteristics {
            name: "Ajani Resolute".into(),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Ajani".into()],
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            loyalty: Some(2),
            text_box: text_box.into(),
            abilities: vec![Ability::when_you_gain_life(Effect::put_counters(
                ObjectSelector::This,
                CounterKind::Loyalty,
                1,
            ))],
            activated_abilities: vec![
                loyalty(0, vec![Effect::gain_life(1)]),
                loyalty(-4, vec![Effect::create_tokens("token.ajanis_pridemate", 1)]),
                loyalty(
                    -10,
                    vec![Effect::CreateEmblem {
                        key: "emblem.ajani_resolute".into(),
                    }],
                ),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ajani_unrelenting() -> CardDefinition {
    let text_box = "Whenever you activate a loyalty ability, create a 2/2 colorless Wizard Soldier creature token named Cadet.\n+1: Creatures you control get +1/+0 and gain haste until end of turn.\n−2: Discard your hand, then draw a card for each creature you control.\n−3: Ajani deals 4 damage to each creature except for tokens you control.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    CardDefinition {
        key: "fra.ajani_unrelenting",
        scryfall_id: "bc3c096f-7f6c-474c-a2c8-75d3e6ddd6f5",
        printed: PrintedCharacteristics {
            name: "Ajani Unrelenting".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Ajani".into()],
            loyalty: Some(5),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActivatesLoyaltyAbility {
                    who: Actor::You,
                    min_counters_removed: 0,
                },
                intervening_if: None,
                effects: vec![Effect::create_tokens("token.cadet", 1)],
            }],
            activated_abilities: vec![
                loyalty(
                    1,
                    Effect::pump_and_grant_until_end_of_turn(
                        ObjectSelector::each_you_control(CardPredicate::creature()),
                        1,
                        0,
                        Keyword::Haste,
                    ),
                ),
                loyalty(
                    -2,
                    vec![
                        Effect::move_to(ObjectSelector::Zone(ZoneType::Hand), ZoneType::Graveyard),
                        Effect::DrawCards {
                            count: ObjectSelector::each_you_control(CardPredicate::creature())
                                .count(),
                        },
                    ],
                ),
                loyalty(
                    -3,
                    vec![Effect::deal_damage(
                        4,
                        ObjectSelector::Each {
                            of: Scope::Battlefield,
                            predicate: CardPredicate::creature(),
                            except: Some(Except::Matching {
                                controlled_by: Actor::You,
                                predicate: Box::new(CardPredicate {
                                    token: Some(true),
                                    ..Default::default()
                                }),
                            }),
                            max_mana_value: None,
                            exact_mana_value: None,
                        },
                    )],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn garruk_curse_breaker() -> CardDefinition {
    let text_box = "Whenever a creature you control with power 4 or greater enters, draw a card.\n+2: Untap up to two target lands.\n−3: Create a 4/4 green Beast creature token with trample.\n−4: Until your next turn, whenever one or more creatures attack one of your opponents, those creatures get +2/+2 and gain trample until end of turn.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    let land_target = TargetSpec::Permanent {
        predicate: CardPredicate::card_type(CardType::Land),
    };
    CardDefinition {
        key: "fra.garruk_curse_breaker",
        scryfall_id: "90ca5812-ceb5-46bd-b049-aed7ff10e6af",
        printed: PrintedCharacteristics {
            name: "Garruk, Curse Breaker".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Garruk".into()],
            loyalty: Some(5),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Creature),
                    min_power: Some(4),
                    ..Default::default()
                }),
                vec![Effect::DrawCards { count: 1.into() }],
            )],
            activated_abilities: vec![
                loyalty(
                    2,
                    vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(2, land_target.clone()),
                        effects: vec![Effect::untap(ObjectSelector::Targets(land_target))],
                    }],
                ),
                loyalty(
                    -3,
                    vec![Effect::create_tokens("token.beast_4_4_g_trample", 1)],
                ),
                loyalty(
                    -4,
                    vec![Effect::CreateDelayedTrigger {
                        event: DelayedEvent::Whenever(Box::new(TriggerCondition::Attacks {
                            subject: TriggerSubject::any(CardPredicate::creature()),
                            per: Per::Batch,
                            attackers: AttackSize::Any,
                            attacked: Actor::Opponent,
                            attacked_kind: AttackedKind::Player,
                            nth: None,
                        })),
                        frequency: DelayedFrequency::EachMatch,
                        expiry: DelayedExpiry::UntilYourNextTurn,
                        remember: None,
                        effects: Effect::pump_and_grant_until_end_of_turn(
                            ObjectSelector::Remembered,
                            2,
                            2,
                            Keyword::Trample,
                        ),
                    }],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn garruk_veiled_butcher() -> CardDefinition {
    let text_box = "If a creature an opponent controls would die, exile it instead.\n+2: Up to one target creature gets -4/-1 until your next turn.\n−2: Each player sacrifices a creature of their choice. If you sacrificed a creature this way, create a 4/4 green Beast creature token with trample.\n−3: Each opponent discards two cards. For each opponent who didn't discard two nonland cards this way, you draw a card.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    CardDefinition {
        key: "fra.garruk_veiled_butcher",
        scryfall_id: "d48bfb8a-d135-45f3-be99-4694b4b9ab93",
        printed: PrintedCharacteristics {
            name: "Garruk, Veiled Butcher".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Garruk".into()],
            loyalty: Some(5),
            replacement_effects: vec![ReplacementEffect::ExileIfWouldDie {
                controlled_by: Actor::Opponent,
                predicate: CardPredicate::creature(),
            }],
            activated_abilities: vec![
                loyalty(
                    2,
                    vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                        effects: vec![Effect::ApplyContinuous {
                            affected: ObjectSelector::Target(TargetSpec::Creature),
                            modification: Modification::pump(-4, -1),
                            duration: EffectDuration::UntilYourNextTurn,
                        }],
                    }],
                ),
                loyalty(
                    -2,
                    vec![
                        Effect::EachPlayerSacrificesCreatures { count: 1 },
                        Effect::if_you_do(vec![Effect::create_tokens(
                            "token.beast_4_4_g_trample",
                            1,
                        )]),
                    ],
                ),
                loyalty(
                    -3,
                    vec![
                        Effect::ForPlayer {
                            player: PlayerSelector::EachOpponent,
                            effects: vec![Effect::Discard {
                                what: ObjectSelector::You,
                                count: 2,
                            }],
                        },
                        Effect::If {
                            condition: Condition::compare(
                                EffectValue::Objects {
                                    of: Box::new(ObjectSelector::CardsThisWay),
                                    predicate: Some(CardPredicate::nonland()),
                                    in_zone: None,
                                },
                                CompareOp::Lt,
                                2,
                            ),
                            then: vec![Effect::DrawCards { count: 1.into() }],
                            otherwise: Vec::new(),
                        },
                    ],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn jace_reality_sculptor() -> CardDefinition {
    let text_box = "+1: Empower Jace X, where X is the number of Islands you control.\n−3: Until your next turn, whenever a creature attacks you or a planeswalker you control, it gets -5/-0 until end of turn.\n0: Exile all but the bottom card of each opponent's library. Activate only if there are twenty-five or more loyalty counters among Jaces you control.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    CardDefinition {
        key: "fra.jace_reality_sculptor",
        scryfall_id: "74087795-0b38-4fd2-9841-147583baca41",
        printed: PrintedCharacteristics {
            name: "Jace, Reality Sculptor".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Jace".into()],
            loyalty: Some(5),
            activated_abilities: vec![
                loyalty(
                    1,
                    Effect::empower_jace(
                        ObjectSelector::each_you_control(CardPredicate::subtype("Island")).count(),
                    ),
                ),
                loyalty(
                    -3,
                    vec![Effect::CreateDelayedTrigger {
                        event: DelayedEvent::Whenever(Box::new(TriggerCondition::Attacks {
                            subject: TriggerSubject::any(CardPredicate::creature()),
                            per: Per::Object,
                            attackers: AttackSize::Any,
                            attacked: Actor::You,
                            attacked_kind: AttackedKind::PlayerOrPlaneswalker,
                            nth: None,
                        })),
                        frequency: DelayedFrequency::EachMatch,
                        expiry: DelayedExpiry::UntilYourNextTurn,
                        remember: None,
                        effects: vec![Effect::until_end_of_turn(
                            ObjectSelector::Triggering,
                            Modification::pump(-5, 0),
                        )],
                    }],
                ),
                ActivatedAbilityDefinition {
                    condition: Some(Condition::compare(
                        ObjectSelector::each_you_control(CardPredicate::subtype("Jace")).property(
                            ObjectProperty::Counters(CounterKind::Loyalty),
                            Aggregate::Sum,
                        ),
                        CompareOp::Ge,
                        25,
                    )),
                    ..loyalty(
                        0,
                        vec![Effect::ForPlayer {
                            player: PlayerSelector::EachOpponent,
                            effects: vec![Effect::exile(ObjectSelector::LibraryExceptBottom)],
                        }],
                    )
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn teyo_enters(keyword: Keyword) -> Ability {
    let spec = TargetSpec::PermanentYouControl {
        predicate: CardPredicate::default(),
    };
    Ability::enters(vec![Effect::TargetClause {
        requirement: TargetRequirement::one(spec.clone()),
        effects: vec![
            Effect::until_end_of_turn(
                ObjectSelector::Target(spec.clone()),
                Modification::AddKeywords(Keywords::of(&[keyword])),
            ),
            Effect::If {
                condition: Condition::TargetMatches(CardPredicate::creature()),
                then: vec![Effect::put_counters(
                    ObjectSelector::Target(spec.clone()),
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
                otherwise: Vec::new(),
            },
            Effect::If {
                condition: Condition::TargetMatches(CardPredicate::card_type(
                    CardType::Planeswalker,
                )),
                then: vec![Effect::put_counters(
                    ObjectSelector::Target(spec),
                    CounterKind::Loyalty,
                    1,
                )],
                otherwise: Vec::new(),
            },
        ],
    }])
}

fn teyo_lightshield_expert() -> CardDefinition {
    let text_box = "Flash\nWhen Teyo enters, target permanent you control gains hexproof until end of turn. Put a +1/+1 counter on it if it's a creature. Put a loyalty counter on it if it's a planeswalker. (It can't be the target of spells or abilities your opponents control.)";
    CardDefinition {
        key: "fra.teyo_lightshield_expert",
        scryfall_id: "5f7521d7-9f1f-4f03-b2ea-dd2a1b1e4e5b",
        printed: PrintedCharacteristics {
            name: "Teyo, Lightshield Expert".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Cleric".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![teyo_enters(Keyword::Hexproof)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn teyo_diamondblade_mage() -> CardDefinition {
    let text_box = "Flash\nWhen Teyo enters, target permanent you control gains deathtouch until end of turn. Put a +1/+1 counter on it if it's a creature. Put a loyalty counter on it if it's a planeswalker.";
    CardDefinition {
        key: "fra.teyo_diamondblade_mage",
        scryfall_id: "100c3b67-0c92-4224-b5ed-67789c612df7",
        printed: PrintedCharacteristics {
            name: "Teyo, Diamondblade Mage".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Warlock".into()],
            power: Some(3),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![teyo_enters(Keyword::Deathtouch)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn your_fate_ends_here() -> CardDefinition {
    let text_box = "Destroy target creature or planeswalker with mana value 3 or greater. Surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "fra.your_fate_ends_here",
        scryfall_id: "25000a17-b701-4d69-b2ef-2c74029199d3",
        printed: PrintedCharacteristics {
            name: "Your Fate Ends Here".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::destroy(ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate {
                        min_mana_value: Some(3),
                        ..CardPredicate::creature_or_planeswalker()
                    },
                })),
                Effect::Surveil { count: 1 },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn terminal_criticism() -> CardDefinition {
    let text_box = "Destroy target creature or planeswalker that's blue or red. You gain 1 life.";
    CardDefinition {
        key: "fra.terminal_criticism",
        scryfall_id: "7ebd7e38-b27c-4c6e-aaea-e8ee5ba5e5df",
        printed: PrintedCharacteristics {
            name: "Terminal Criticism".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::destroy(ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate {
                        colors_any: vec![Color::Blue, Color::Red],
                        ..CardPredicate::creature_or_planeswalker()
                    },
                })),
                Effect::gain_life(1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn refute_destiny() -> CardDefinition {
    let text_box = "Exile target creature or planeswalker that's green or blue. Surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "fra.refute_destiny",
        scryfall_id: "2c588954-c6eb-4aae-a2fa-0651ccf2d90a",
        printed: PrintedCharacteristics {
            name: "Refute Destiny".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::exile(ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate {
                        colors_any: vec![Color::Green, Color::Blue],
                        ..CardPredicate::creature_or_planeswalker()
                    },
                })),
                Effect::Surveil { count: 1 },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn extended_absence() -> CardDefinition {
    let text_box = "Exile target creature or planeswalker. Extended Absence deals 1 damage to each opponent and you gain 1 life.";
    CardDefinition {
        key: "fra.extended_absence",
        scryfall_id: "eb4b6ed8-782e-4473-abc9-d50bf2275c6a",
        printed: PrintedCharacteristics {
            name: "Extended Absence".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::exile(ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate::creature_or_planeswalker(),
                })),
                Effect::DealDamageToPlayer {
                    player: PlayerSelector::EachOpponent,
                    amount: 1.into(),
                },
                Effect::gain_life(1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fulminous_forte() -> CardDefinition {
    let text_box = "Choose one —\n• Fulminous Forte deals 1 damage to each creature and planeswalker your opponents control.\n• Fulminous Forte deals 5 damage to target creature or planeswalker.";
    CardDefinition {
        key: "fra.fulminous_forte",
        scryfall_id: "19acb2b5-3b3e-43f0-bd81-8426ed3d9c55",
        printed: PrintedCharacteristics {
            name: "Fulminous Forte".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::deal_damage(
                        1,
                        ObjectSelector::each(
                            Scope::OpponentsControl,
                            CardPredicate::creature_or_planeswalker(),
                        ),
                    )],
                    vec![Effect::deal_damage(
                        5,
                        ObjectSelector::Target(TargetSpec::Permanent {
                            predicate: CardPredicate::creature_or_planeswalker(),
                        }),
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn essence_burn() -> CardDefinition {
    let text_box = "Essence Burn deals 5 damage to target black or green creature or planeswalker. If that permanent would die this turn, exile it instead.";
    let spec = TargetSpec::Permanent {
        predicate: CardPredicate {
            colors_any: vec![Color::Black, Color::Green],
            ..CardPredicate::creature_or_planeswalker()
        },
    };
    CardDefinition {
        key: "fra.essence_burn",
        scryfall_id: "d2e958de-70de-4156-8f9b-b2c0c1ba704a",
        printed: PrintedCharacteristics {
            name: "Essence Burn".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::TargetClause {
                requirement: TargetRequirement::one(spec.clone()),
                effects: vec![
                    Effect::deal_damage(5, ObjectSelector::Target(spec.clone())),
                    Effect::until_end_of_turn(
                        ObjectSelector::Target(spec),
                        Modification::ExileIfWouldDie,
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wrath_of_the_bloodmane() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast if you control a legendary creature.\nWrath of the Bloodmane deals 4 damage to target creature or planeswalker.";
    CardDefinition {
        key: "fra.wrath_of_the_bloodmane",
        scryfall_id: "b5b55617-684a-4036-be9b-a3b24fc9cd5a",
        printed: PrintedCharacteristics {
            name: "Wrath of the Bloodmane".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::CostsLess {
                amount: 1.into(),
                condition: Some(Condition::ControlsAtLeast {
                    distinct_names: false,
                    count: 1,
                    predicate: CardPredicate {
                        supertype: Some("Legendary".into()),
                        ..CardPredicate::creature()
                    },
                }),
            }],
            spell_effects: vec![Effect::deal_damage(
                4,
                ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate::creature_or_planeswalker(),
                }),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn silence_the_echo() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, sacrifice a creature or planeswalker or pay {3}.\nDestroy target creature or planeswalker.";
    CardDefinition {
        key: "fra.silence_the_echo",
        scryfall_id: "f1d274db-751b-4414-a38d-762198168e91",
        printed: PrintedCharacteristics {
            name: "Silence the Echo".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::Sacrifice(Box::new(SacrificeCost {
                predicate: CardPredicate::creature_or_planeswalker(),
                another: false,
                or_mana: Some(ManaCost::parse("{3}")),
            }))],
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate::creature_or_planeswalker(),
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn break_under_pressure() -> CardDefinition {
    let text_box = "Target opponent sacrifices a creature or planeswalker with the greatest mana value among creatures and planeswalkers they control. You gain 2 life.";
    let max_mv = ObjectSelector::each_you_control(CardPredicate::creature_or_planeswalker())
        .property(ObjectProperty::ManaValue, Aggregate::Max);
    CardDefinition {
        key: "fra.break_under_pressure",
        scryfall_id: "46974d94-e900-43e4-92b5-4fb9b9f7cf46",
        printed: PrintedCharacteristics {
            name: "Break Under Pressure".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::ForPlayer {
                    player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                    effects: vec![Effect::SelectCards {
                        from: ObjectSelector::Each {
                            of: Scope::YouControl,
                            predicate: CardPredicate::creature_or_planeswalker(),
                            except: None,
                            max_mana_value: None,
                            exact_mana_value: Some(Box::new(max_mv)),
                        },
                        predicate: CardPredicate::default(),
                        min: 1,
                        max: 1.into(),
                        effects: vec![Effect::Sacrifice {
                            what: ObjectSelector::Selected,
                            optional: false,
                        }],
                        if_declined: Vec::new(),
                    }],
                },
                Effect::gain_life(2),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn awaken_the_inferno() -> CardDefinition {
    let text_box = "Awaken the Inferno deals 6 damage to target creature or planeswalker an opponent controls. Put a +1/+1 counter on up to one target creature you control.\nBasic landcycling {2} ({2}, Discard this card: Search your library for a basic land card, reveal it, put it into your hand, then shuffle.)";
    let opp_target = TargetSpec::PermanentOpponentControls {
        predicate: CardPredicate::creature_or_planeswalker(),
    };
    CardDefinition {
        key: "fra.awaken_the_inferno",
        scryfall_id: "c596c4ec-8480-4be9-a45d-700398a126f6",
        printed: PrintedCharacteristics {
            name: "Awaken the Inferno".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::TargetClause {
                    requirement: TargetRequirement::one(opp_target.clone()),
                    effects: vec![Effect::deal_damage(6, ObjectSelector::Target(opp_target))],
                },
                Effect::TargetClause {
                    requirement: TargetRequirement::up_to(1, TargetSpec::CreatureYouControl),
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                zone: ActivationZone::Hand,
                discard_source: true,
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::search_library(
                    CardPredicate::basic_land(),
                    1.into(),
                    vec![Effect::return_to_hand(ObjectSelector::Selected)],
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn craftwork_crusher() -> CardDefinition {
    let text_box = "Trample\nWhen this creature enters, choose two —\n• This creature deals 4 damage to target creature or planeswalker.\n• Create a 2/2 colorless Wizard Soldier creature token named Cadet.\n• Draw a card.";
    CardDefinition {
        key: "fra.craftwork_crusher",
        scryfall_id: "03f9839c-aa07-4ee7-847b-091e47ab80c4",
        printed: PrintedCharacteristics {
            name: "Craftwork Crusher".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}{G}{G}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Boar".into(), "Construct".into()],
            power: Some(7),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::enters(vec![Effect::TriggerModes {
                modes: vec![
                    vec![Effect::deal_damage(
                        4,
                        ObjectSelector::Target(TargetSpec::Permanent {
                            predicate: CardPredicate::creature_or_planeswalker(),
                        }),
                    )],
                    vec![Effect::create_tokens("token.cadet", 1)],
                    vec![Effect::DrawCards { count: 1.into() }],
                ],
                min: 2,
                up_to: Some(2.into()),
                unchosen: false,
            }])],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn compel_brutality() -> CardDefinition {
    let text_box = "Choose one —\n• Target creature you control deals damage equal to its power to target creature or planeswalker an opponent controls.\n• Target planeswalker you control deals damage equal to its loyalty to target creature or planeswalker an opponent controls.";
    let opp_target = TargetSpec::PermanentOpponentControls {
        predicate: CardPredicate::creature_or_planeswalker(),
    };
    let own_pw = TargetSpec::PermanentYouControl {
        predicate: CardPredicate::card_type(CardType::Planeswalker),
    };
    let bite_mode = |source_spec: TargetSpec, property: ObjectProperty| {
        vec![
            Effect::TargetClause {
                requirement: TargetRequirement::one(source_spec.clone()),
                effects: vec![Effect::Remember {
                    what: ObjectSelector::Target(source_spec),
                }],
            },
            Effect::TargetClause {
                requirement: TargetRequirement::one(opp_target.clone()),
                effects: vec![Effect::If {
                    condition: Condition::compare(
                        ObjectSelector::Remembered.count(),
                        CompareOp::Ge,
                        1,
                    ),
                    then: vec![Effect::DealDamage {
                        amount: ObjectSelector::Remembered.property(property, Aggregate::Sum),
                        what: ObjectSelector::Target(opp_target.clone()),
                        by: Some(ObjectSelector::Remembered),
                    }],
                    otherwise: Vec::new(),
                }],
            },
        ]
    };
    CardDefinition {
        key: "fra.compel_brutality",
        scryfall_id: "bd32d736-7a58-46b9-90b4-2cac3c3e80a1",
        printed: PrintedCharacteristics {
            name: "Compel Brutality".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    bite_mode(TargetSpec::CreatureYouControl, ObjectProperty::Power),
                    bite_mode(own_pw, ObjectProperty::Counters(CounterKind::Loyalty)),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn edgar_ancient_bloodlord() -> CardDefinition {
    let text_box = "Whenever another creature or planeswalker you control dies, you gain 1 life.\n{2}, Sacrifice another creature or planeswalker: Put a +1/+1 counter on Edgar. He gains menace until end of turn. (He can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "fra.edgar_ancient_bloodlord",
        scryfall_id: "7c619fed-2394-4efc-8cdc-6df5f51c1f57",
        printed: PrintedCharacteristics {
            name: "Edgar, Ancient Bloodlord".into(),
            mana_cost: Some(ManaCost::parse("{W}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Vampire".into(), "Noble".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::another_you_control(
                        CardPredicate::creature_or_planeswalker(),
                    ),
                },
                intervening_if: None,
                effects: vec![Effect::gain_life(1)],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate::creature_or_planeswalker(),
                    another: true,
                    or_mana: None,
                }),
                effects: vec![
                    Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 1),
                    Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                    ),
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn liliana_the_faultless() -> CardDefinition {
    let text_box = "Whenever another creature or planeswalker you control enters, you gain 1 life.\n{1}, {T}, Discard a card: Another target creature or planeswalker you control gains hexproof until end of turn.";
    CardDefinition {
        key: "fra.liliana_the_faultless",
        scryfall_id: "70d8c400-87dc-4f15-808f-e54a95d779fc",
        printed: PrintedCharacteristics {
            name: "Liliana the Faultless".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Cleric".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate::creature_or_planeswalker()),
                vec![Effect::gain_life(1)],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                tap_source: true,
                discard_card: true,
                effects: vec![Effect::another_target(
                    TargetSpec::PermanentYouControl {
                        predicate: CardPredicate::creature_or_planeswalker(),
                    },
                    |spec| {
                        vec![Effect::until_end_of_turn(
                            ObjectSelector::Target(spec),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Hexproof])),
                        )]
                    },
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn liliana_the_repentant() -> CardDefinition {
    let text_box = "Whenever another creature or planeswalker you control enters, mill two cards.\nExhaust — {5}{B}: Return target creature or planeswalker card from your graveyard to the battlefield. Put a +1/+1 counter on Liliana. Activate only as a sorcery. (Activate each exhaust ability only once.)";
    CardDefinition {
        key: "fra.liliana_the_repentant",
        scryfall_id: "1eb25a6c-d6b4-465d-990e-f1ab86b26b69",
        printed: PrintedCharacteristics {
            name: "Liliana the Repentant".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Warlock".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate::creature_or_planeswalker()),
                vec![Effect::MillCards {
                    count: 2.into(),
                    target: None,
                    player: None,
                }],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition {
                exhaust: true,
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{5}{B}"),
                effects: vec![
                    Effect::return_to_battlefield(
                        ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                            predicate: CardPredicate::creature_or_planeswalker(),
                        }),
                        false,
                    ),
                    Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 1),
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn massacre_girl_most_wanted() -> CardDefinition {
    let text_box = "Whenever another creature or planeswalker you control dies, Massacre Girl deals 1 damage to target opponent and you gain 1 life.\nWhenever an opponent is dealt noncombat damage, put a +1/+1 counter on Massacre Girl.";
    CardDefinition {
        key: "fra.massacre_girl_most_wanted",
        scryfall_id: "9028d31f-9c41-47e3-885b-6a869bca8178",
        printed: PrintedCharacteristics {
            name: "Massacre Girl, Most Wanted".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Assassin".into()],
            power: Some(4),
            toughness: Some(4),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::another_you_control(
                            CardPredicate::creature_or_planeswalker(),
                        ),
                    },
                    intervening_if: None,
                    effects: vec![
                        Effect::DealDamageToPlayer {
                            player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                            amount: 1.into(),
                        },
                        Effect::gain_life(1),
                    ],
                },
                Ability::Triggered {
                    condition: TriggerCondition::DealsDamage {
                        kind: DamageKind::Noncombat,
                        source: TriggerSubject::any(CardPredicate::default()),
                        to: DamageRecipientFilter::Player(Actor::Opponent),
                    },
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn winter_tormented_loner() -> CardDefinition {
    let text_box = "When Winter enters, you may sacrifice a creature or planeswalker. When you do, each opponent sacrifices a creature of their choice.\nWinter gets +1/+0 for each creature and planeswalker card in your graveyard.";
    CardDefinition {
        key: "fra.winter_tormented_loner",
        scryfall_id: "9670f754-f41f-45ac-8e8b-025ad2c0f66b",
        printed: PrintedCharacteristics {
            name: "Winter, Tormented Loner".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Warlock".into()],
            power: Some(0),
            toughness: Some(3),
            abilities: vec![Ability::enters(vec![
                Effect::may_sacrifice(CardPredicate::creature_or_planeswalker(), false),
                Effect::if_you_do(vec![Effect::WhenYouDo {
                    effects: vec![Effect::ForPlayer {
                        player: PlayerSelector::EachOpponent,
                        effects: vec![Effect::sacrifice(CardPredicate::creature())],
                    }],
                }]),
            ])],
            statics: vec![StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::This,
                condition: None,
                power: 1,
                toughness: 0,
                per_creature_type_max: None,
                per_color_among: None,
                per_graveyard_matching: Some(CardPredicate::creature_or_planeswalker()),
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mabel_bitter_recluse() -> CardDefinition {
    let text_box = "Deathtouch\nWhen Mabel enters, remove up to three counters from another target creature or planeswalker.";
    CardDefinition {
        key: "fra.mabel_bitter_recluse",
        scryfall_id: "b2a412b0-2ae4-4552-bc5e-70654b6b9b4e",
        printed: PrintedCharacteristics {
            name: "Mabel, Bitter Recluse".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Mouse".into(), "Warlock".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            abilities: vec![Ability::enters(vec![Effect::another_target(
                TargetSpec::Permanent {
                    predicate: CardPredicate::creature_or_planeswalker(),
                },
                |spec| {
                    vec![Effect::RemoveCounters {
                        from: ObjectSelector::Target(spec),
                        up_to: 3,
                        chosen: None,
                    }]
                },
            )])],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn living_library() -> CardDefinition {
    let text_box = "{6}, Sacrifice this creature: Choose target creature or planeswalker an opponent controls. Its owner shuffles it into their library.";
    CardDefinition {
        key: "fra.living_library",
        scryfall_id: "5d4a8e5f-0024-4da3-a2f5-edb48b12e733",
        printed: PrintedCharacteristics {
            name: "Living Library".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Book".into(), "Illusion".into()],
            power: Some(0),
            toughness: Some(4),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{6}"),
                sacrifice_source: true,
                effects: vec![Effect::move_to(
                    ObjectSelector::Target(TargetSpec::PermanentOpponentControls {
                        predicate: CardPredicate::creature_or_planeswalker(),
                    }),
                    ZoneType::Library,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn identity_echo() -> CardDefinition {
    let text_box = "{3}{R}: Exile target creature or planeswalker you control. Reveal cards from the top of your library until you reveal a creature or planeswalker card. Put that card onto the battlefield and the rest on the bottom of your library in a random order. Activate only as a sorcery.";
    CardDefinition {
        key: "fra.identity_echo",
        scryfall_id: "e600b33b-8916-43dd-95d3-d7cbf874933d",
        printed: PrintedCharacteristics {
            name: "Identity Echo".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{R}"),
                sorcery_timing: true,
                effects: vec![
                    Effect::exile(ObjectSelector::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate::creature_or_planeswalker(),
                    })),
                    Effect::RevealUntil {
                        predicate: CardPredicate::creature_or_planeswalker(),
                        to: ZoneType::Battlefield,
                    },
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vindictive_triumph() -> CardDefinition {
    let text_box = "Exile target creature or planeswalker. If that permanent's mana value was 3 or less, return it to the battlefield tapped under your control. Exile it at the beginning of the next end step.";
    let spec = TargetSpec::Permanent {
        predicate: CardPredicate::creature_or_planeswalker(),
    };
    CardDefinition {
        key: "fra.vindictive_triumph",
        scryfall_id: "a803dbe7-153a-4e92-ad4d-c2babebe003d",
        printed: PrintedCharacteristics {
            name: "Vindictive Triumph".into(),
            mana_cost: Some(ManaCost::parse("{W}{B}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::TargetClause {
                requirement: TargetRequirement::one(spec.clone()),
                effects: vec![
                    Effect::exile(ObjectSelector::Target(spec.clone())),
                    Effect::If {
                        condition: Condition::compare(
                            ObjectSelector::Target(spec)
                                .property(ObjectProperty::ManaValue, Aggregate::Sum),
                            CompareOp::Le,
                            3,
                        ),
                        then: vec![
                            Effect::return_to_battlefield(ObjectSelector::Moved, true),
                            Effect::at_next_end_step(
                                ObjectSelector::Moved,
                                vec![Effect::exile(ObjectSelector::Remembered)],
                            ),
                        ],
                        otherwise: Vec::new(),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn entrust_the_spark() -> CardDefinition {
    let text_box = "You may sacrifice a planeswalker. If you do, search your library for a planeswalker card, put it onto the battlefield, then shuffle.";
    let pw = CardPredicate::card_type(CardType::Planeswalker);
    CardDefinition {
        key: "fra.entrust_the_spark",
        scryfall_id: "ca894d25-b9fc-4cd6-8746-70d8c2868721",
        printed: PrintedCharacteristics {
            name: "Entrust the Spark".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::may_sacrifice(pw.clone(), false),
                Effect::if_you_do(vec![Effect::search_library(
                    pw,
                    1.into(),
                    vec![Effect::return_to_battlefield(
                        ObjectSelector::Selected,
                        false,
                    )],
                )]),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn loyal_tutor() -> CardDefinition {
    let text_box = "Search your library for a planeswalker card, reveal it, then shuffle and put that card on top.";
    CardDefinition {
        key: "fra.loyal_tutor",
        scryfall_id: "490dae91-94ce-42a9-a11f-6c5e77c4e486",
        printed: PrintedCharacteristics {
            name: "Loyal Tutor".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::SelectCards {
                from: ObjectSelector::Zone(ZoneType::Library),
                predicate: CardPredicate::card_type(CardType::Planeswalker),
                min: 0,
                max: 1.into(),
                effects: vec![
                    Effect::Reveal {
                        what: ObjectSelector::Selected,
                    },
                    Effect::ShuffleLibrary,
                    Effect::put_in_library(ObjectSelector::Selected, LibraryEnd::Top),
                ],
                if_declined: vec![Effect::ShuffleLibrary],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mind_meanderer() -> CardDefinition {
    let text_box = "Flying\nThis creature has vigilance as long as you control a Jace planeswalker.\nWhen this creature enters, it fights up to one target creature an opponent controls. (Each deals damage equal to its power to the other.)";
    CardDefinition {
        key: "fra.mind_meanderer",
        scryfall_id: "94c290ce-252c-42b3-bcb0-c1ef621df566",
        printed: PrintedCharacteristics {
            name: "Mind Meanderer".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into(), "Fish".into(), "Illusion".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::This,
                condition: Some(Condition::ControlsAtLeast {
                    distinct_names: false,
                    count: 1,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Planeswalker),
                        subtype: Some("Jace".into()),
                        ..Default::default()
                    },
                }),
                keywords: Keywords::of(&[Keyword::Vigilance]),
            })],
            abilities: vec![Ability::enters(vec![Effect::TargetClause {
                requirement: TargetRequirement::up_to(1, TargetSpec::CreatureOpponentControls),
                effects: vec![Effect::Fight {
                    first: ObjectSelector::ExactSource,
                    second: ObjectSelector::Target(TargetSpec::CreatureOpponentControls),
                }],
            }])],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lichs_relic() -> CardDefinition {
    let text_box = "When this Equipment enters, you may pay {2}. When you do, for each opponent, destroy up to one target creature or planeswalker that player controls.\nEquipped creature gets +2/+1.\nEquip {2}";
    let opp_target = TargetSpec::PermanentOpponentControls {
        predicate: CardPredicate::creature_or_planeswalker(),
    };
    CardDefinition {
        key: "fra.lichs_relic",
        scryfall_id: "b105511d-5022-4a84-b6ce-4bb433e93a62",
        printed: PrintedCharacteristics {
            name: "Lich's Relic".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            abilities: vec![Ability::enters(vec![Effect::OptionalPayment {
                cost: PaymentCost::Mana(ManaCost::parse("{2}")),
                if_paid: vec![Effect::WhenYouDo {
                    effects: vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, opp_target.clone()),
                        effects: vec![Effect::destroy(ObjectSelector::Target(opp_target))],
                    }],
                }],
                if_declined: Vec::new(),
            }])],
            statics: vec![StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::Attached,
                condition: None,
                power: 2,
                toughness: 1,
                per_creature_type_max: None,
                per_color_among: None,
                per_graveyard_matching: None,
            })],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn face_yourself() -> CardDefinition {
    let text_box = "For each creature target player controls, create a token that's a copy of that creature, except it has haste and \"At the beginning of the end step, if you don't control a planeswalker, sacrifice this token.\"";
    CardDefinition {
        key: "fra.face_yourself",
        scryfall_id: "3ccf8f64-19bd-4fdf-b70a-30a042bacf2f",
        printed: PrintedCharacteristics {
            name: "Face Yourself".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::CreateTokenCopy {
                count: 1.into(),
                of: ObjectSelector::each(Scope::TargetPlayerControls, CardPredicate::creature()),
                exceptions: vec![
                    CopyException::AddKeywords(Keywords::of(&[Keyword::Haste])),
                    CopyException::GrantTriggered(Box::new(Ability::Triggered {
                        condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                        intervening_if: Some(Condition::not(Condition::ControlsAtLeast {
                            distinct_names: false,
                            count: 1,
                            predicate: CardPredicate::card_type(CardType::Planeswalker),
                        })),
                        effects: vec![Effect::sacrifice_this()],
                    })),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
