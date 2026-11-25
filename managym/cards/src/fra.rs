//! Reality Fracture.
use super::enters_prepared;
use managym_state::{
    ability::{
        Ability, Actor, CompareOp, Condition, Effect, EffectValue, KeywordAction, LibraryEnd,
        ObjectSelector, Per, PlayerSelector, Scope, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, ActivationZone, CardDefinition, CardType, CardTypes,
        InsetSpell, Keyword, Keywords, LayerValue, PrintedCharacteristics, ReplacementEffect,
        Restriction, StaticAbility, StaticContinuous, StaticRestriction, StaticScope,
    },
    effect::Modification,
    mana::{Mana, ManaCost, ManaRestriction},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
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
    use managym_state::card::LoyaltyCost;
    let text_box = "At the beginning of each opponent's draw step, you draw a card.\n+1: Create a 1/1 blue Illusion creature token.\n−2: For each opponent, return up to one target artifact or creature that player controls to its owner's hand.\n−6: Draw three cards. Then put X +1/+1 counters on each creature you control, where X is the number of cards in your hand.";
    let bounce = TargetSpec::PermanentOpponentControls {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Artifact, CardType::Creature],
            ..Default::default()
        },
    };
    let loyalty = |cost, effects| ActivatedAbilityDefinition {
        loyalty: Some(LoyaltyCost::Fixed(cost)),
        effects,
        ..Default::default()
    };
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
                    CardPredicate {
                        card_type: Some(CardType::Land),
                        supertype: Some("Basic".into()),
                        ..Default::default()
                    },
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

fn way_of_the_mentor() -> CardDefinition {
    let text_box = "When Way of the Mentor enters, empower Jace 5. (Put five loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\nWhenever you gain life, put a loyalty counter on each planeswalker you control.";
    CardDefinition {
        key: "fra.way_of_the_mentor",
        scryfall_id: "1a59d5b1-12d6-486b-bd29-ca371359addd",
        printed: PrintedCharacteristics {
            name: "Way of the Mentor".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            supertypes: vec!["Legendary".into()],
            text_box: text_box.into(),
            abilities: vec![
                Ability::enters(Effect::empower_jace(5)),
                Ability::when_you_gain_life(Effect::PutCounters {
                    on: ObjectSelector::each_you_control(CardPredicate {
                        card_type: Some(CardType::Planeswalker),
                        ..Default::default()
                    }),
                    kind: CounterKind::Loyalty,
                    amount: 1.into(),
                }),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn way_of_the_necromancer() -> CardDefinition {
    let text_box = "When Way of the Necromancer enters, empower Jace 2. (Put two loyalty counters on a Jace token you control. If you don't control one, first create a blue Jace planeswalker token with \"[−1]: Surveil 1\" and \"[−3]: Draw a card.\")\nWhenever a creature you control dies, put a loyalty counter on each planeswalker you control.";
    CardDefinition {
        key: "fra.way_of_the_necromancer",
        scryfall_id: "a0ff9689-ea49-4fff-b37c-4abbaeb0f73d",
        printed: PrintedCharacteristics {
            name: "Way of the Necromancer".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            supertypes: vec!["Legendary".into()],
            text_box: text_box.into(),
            abilities: vec![
                Ability::enters(Effect::empower_jace(2)),
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::you_control(CardPredicate::creature()),
                    },
                    intervening_if: None,
                    effects: vec![Effect::PutCounters {
                        on: ObjectSelector::each_you_control(CardPredicate {
                            card_type: Some(CardType::Planeswalker),
                            ..Default::default()
                        }),
                        kind: CounterKind::Loyalty,
                        amount: 1.into(),
                    }],
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}
