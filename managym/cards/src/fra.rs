//! Reality Fracture.
use super::enters_prepared;
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, KeywordAction, LibraryEnd, ObjectSelector, Per,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, InsetSpell, Keyword,
        Keywords, LayerValue, PrintedCharacteristics, ReplacementEffect, Restriction,
        StaticAbility, StaticContinuous, StaticRestriction, StaticScope,
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
    use managym_state::{ability::EffectValue, card::LoyaltyCost};
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
