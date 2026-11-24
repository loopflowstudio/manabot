//! Final Fantasy.
use managym_state::{
    ability::{
        Ability, Actor, Condition, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect,
        EffectValue, Except, ObjectSelector, PaymentCost, Per, PlayerSelector, Scope,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, SpellCostReduction, StaticAbility, StaticKeywordGrant, StaticScope,
    },
    effect::Modification,
    mana::{Color, Colors, Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        the_fire_crystal(),
        cecil(),
        esper_origins(),
        fang(),
        vanille(),
        sazh_s_chocobo(),
    ]
}

fn cecil() -> CardDefinition {
    let text_box = "Deathtouch\nDarkness — Whenever Cecil deals damage, you lose that much life. Then if your life total is less than or equal to half your starting life total, untap Cecil and transform it.";
    let back_text = "Lifelink\nProtect — Whenever Cecil attacks, other attacking creatures gain indestructible until end of turn.";
    let creature = PrintedCharacteristics {
        supertypes: vec!["Legendary".into()],
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Knight".into()],
        ..Default::default()
    };
    CardDefinition {
        key: "fin.cecil_dark_knight",
        scryfall_id: "026e7167-d665-43d0-a51e-8df2d68cdb5e",
        printed: PrintedCharacteristics {
            name: "Cecil, Dark Knight".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::SourceDealsDamage,
                intervening_if: None,
                effects: vec![
                    Effect::LoseLife {
                        player: PlayerSelector::You,
                        amount: EffectValue::TriggerDamage,
                    },
                    Effect::If {
                        condition: Condition::AtMostHalfStartingLife,
                        then: vec![
                            Effect::Untap {
                                what: ObjectSelector::This,
                            },
                            Effect::TransformSource,
                        ],
                        otherwise: vec![],
                    },
                ],
            }],
            text_box: text_box.into(),
            ..creature.clone()
        },
        back_face: Some(Box::new(PrintedCharacteristics {
            name: "Cecil, Redeemed Paladin".into(),
            power: Some(4),
            toughness: Some(4),
            color_override: Some(Colors::from_iter([Color::White])),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::Each {
                        of: Scope::Battlefield,
                        predicate: CardPredicate {
                            attacking: true,
                            ..CardPredicate::creature()
                        },
                        except: Some(Except::This),
                        max_mana_value: None,
                    },
                    Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                )],
            }],
            text_box: back_text.into(),
            ..creature
        })),
    }
}

fn fang() -> CardDefinition {
    CardDefinition {
        key: "fin.fang_fearless_lcie",
        scryfall_id: "f73a5cbb-905f-4b99-8d23-9d1b6bd47ee8",
        printed: PrintedCharacteristics {
            name: FANG.into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            supertypes: vec!["Legendary".into()],
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::OnceEachTurn { condition: Box::new(TriggerCondition::CardsLeaveYourGraveyard) },
                intervening_if: None,
                effects: vec![Effect::DrawCards { count: EffectValue::Fixed(1) }, Effect::LoseLife { player: PlayerSelector::You, amount: EffectValue::Fixed(1) }],
            }],
            text_box: "Whenever one or more cards leave your graveyard, you draw a card and you lose 1 life. This ability triggers only once each turn.\n(Melds with Vanille, Cheerful l'Cie.)".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

const VANILLE: &str = "Vanille, Cheerful l'Cie";
const FANG: &str = "Fang, Fearless l'Cie";
const RAGNAROK: &str = "Ragnarok, Divine Deliverance";

fn vanille() -> CardDefinition {
    let text_box = "When Vanille enters, mill two cards, then return a permanent card from your graveyard to your hand.\nAt the beginning of your first main phase, if you both own and control Vanille and a creature named Fang, Fearless l'Cie, you may pay {3}{B}{G}. If you do, exile them, then meld them into Ragnarok, Divine Deliverance.";
    CardDefinition {
        key: "fin.vanille_cheerful_lcie",
        scryfall_id: "91226c1a-63a0-494e-bcf0-77c2d6f49213",
        printed: PrintedCharacteristics {
            name: VANILLE.into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            supertypes: vec!["Legendary".into()],
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Cleric".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![
                        Effect::MillCards {
                            count: 2.into(),
                            target: None,
                        },
                        Effect::SelectCards {
                            from: ObjectSelector::Zone(ZoneType::Graveyard),
                            predicate: CardPredicate::permanent(),
                            min: 1,
                            max: 1.into(),
                            effects: vec![Effect::return_to_hand(ObjectSelector::Selected)],
                            if_declined: vec![],
                        },
                    ],
                },
                Ability::Triggered {
                    condition: TriggerCondition::beginning_of(TurnStep::FirstMain, Actor::You),
                    intervening_if: Some(Condition::OwnsMeldPair {
                        partner: FANG.into(),
                    }),
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::parse("{3}{B}{G}")),
                        if_paid: vec![
                            Effect::ExileMeldPair {
                                partner: FANG.into(),
                            },
                            Effect::MeldExiled {
                                fronts: [VANILLE.into(), FANG.into()],
                                result: RAGNAROK.into(),
                            },
                        ],
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

/// The single face shown by the two melded cards (CR 713).
pub fn ragnarok() -> managym_state::card::ObjectDefinition {
    let text_box = "Reach, vigilance, menace, trample, haste\nWhen Ragnarok dies, destroy target permanent and return target nonlegendary permanent card from your graveyard to the battlefield.";
    let destroy = TargetSpec::Permanent {
        predicate: CardPredicate::default(),
    };
    let reanimate = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate {
            nonlegendary: true,
            ..CardPredicate::permanent()
        },
    };
    let face = PrintedCharacteristics {
        name: RAGNAROK.into(),
        supertypes: vec!["Legendary".into()],
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Beast".into(), "Avatar".into()],
        power: Some(7),
        toughness: Some(6),
        color_override: Some(Colors::from_iter([Color::Black, Color::Green])),
        keywords: Keywords::of(&[
            Keyword::Vigilance,
            Keyword::Menace,
            Keyword::Trample,
            Keyword::Reach,
            Keyword::Haste,
        ]),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            intervening_if: None,
            effects: vec![
                Effect::TargetClause {
                    requirement: TargetRequirement::one(destroy.clone()),
                    effects: vec![Effect::Destroy {
                        what: ObjectSelector::Target(destroy),
                    }],
                },
                Effect::TargetClause {
                    requirement: TargetRequirement::one(reanimate.clone()),
                    effects: vec![Effect::move_to(
                        ObjectSelector::Target(reanimate),
                        ZoneType::Battlefield,
                    )],
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    };
    managym_state::card::ObjectDefinition {
        scryfall_id: Some("a951cc35-bd29-467f-b28c-5d57340e7450"),
        ..managym_state::card::ObjectDefinition::face(face)
    }
}

fn esper_origins() -> CardDefinition {
    let text_box = "Surveil 2. You gain 2 life. If this spell was cast from a graveyard, exile it, then put it onto the battlefield transformed under its owner's control with a finality counter on it. (If a creature with a finality counter on it would die, exile it instead.)\nFlashback {3}{G} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    let back_text = "(As this Saga enters and after your draw step, add a lore counter. Sacrifice after III.)\nI — Reveal the top card of your library. If it's a permanent card, put it into your hand.\nII — Add {G}{G}.\nIII — Other creatures you control get +2/+2 and gain trample until end of turn.";
    let other_creatures = ObjectSelector::each_other_you_control(CardPredicate::creature());
    CardDefinition {
        key: "fin.esper_origins",
        scryfall_id: "0f503360-216a-4629-89b2-d32072850aef",
        printed: PrintedCharacteristics {
            name: "Esper Origins".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::Surveil { count: 2 },
                Effect::gain_life(2),
                Effect::If {
                    condition: Condition::CastFromGraveyard,
                    then: Effect::exile_and_return_transformed(
                        Some(PlayerSelector::Owner),
                        Some(CounterKind::Finality),
                    ),
                    otherwise: vec![],
                },
            ],
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{3}{G}"))],
            text_box: text_box.into(),
            ..Default::default()
        },
        back_face: Some(Box::new(PrintedCharacteristics {
            name: "Summon: Esper Maduin".into(),
            types: CardTypes::new([CardType::Enchantment, CardType::Creature]),
            subtypes: vec!["Saga".into(), "Elemental".into()],
            color_override: Some(Colors::from_iter([Color::Green])),
            power: Some(4),
            toughness: Some(4),
            abilities: vec![
                Ability::chapter(
                    1,
                    vec![
                        Effect::RevealLibraryTop,
                        Effect::If {
                            condition: Condition::CardsThisWayMatch(CardPredicate::permanent()),
                            then: vec![Effect::PutTopCardsInHand { count: 1 }],
                            otherwise: vec![],
                        },
                    ],
                ),
                Ability::chapter(2, vec![Effect::add_mana(Mana::parse("GG"))]),
                Ability::chapter(
                    3,
                    vec![
                        Effect::until_end_of_turn(
                            other_creatures.clone(),
                            Modification::pump(2, 2),
                        ),
                        Effect::until_end_of_turn(
                            other_creatures,
                            Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                        ),
                    ],
                ),
            ],
            text_box: back_text.into(),
            ..Default::default()
        })),
    }
}

fn sazh_s_chocobo() -> CardDefinition {
    let text_box =
        "Landfall — Whenever a land you control enters, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "fin.sazh_s_chocobo",
        scryfall_id: "dda6b4d0-1b60-46b0-b321-b9ffe15afff4",
        printed: PrintedCharacteristics {
            name: "Sazh's Chocobo".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(0),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
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

fn the_fire_crystal() -> CardDefinition {
    let text_box = "Red spells you cast cost {1} less to cast.\nCreatures you control have haste.\n{4}{R}{R}, {T}: Create a token that's a copy of target creature you control. Sacrifice it at the beginning of the next end step.";
    CardDefinition {
        key: "fin.the_fire_crystal",
        scryfall_id: "ea430b17-2014-4b8e-b53f-43bcfc06f7cd",
        printed: PrintedCharacteristics {
            name: "The Fire Crystal".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            statics: vec![
                StaticAbility::SpellsCostLess(SpellCostReduction {
                    first_on_own_turn: false,
                    predicate: CardPredicate {
                        colors_any: vec![Color::Red],
                        ..Default::default()
                    },
                    condition: Condition::Always,
                    generic: 1,
                }),
                StaticAbility::KeywordGrant(StaticKeywordGrant {
                    scope: StaticScope::YouControl(CardPredicate::creature()),
                    condition: None,
                    keywords: Keywords::of(&[Keyword::Haste]),
                }),
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}{R}{R}"),
                tap_source: true,
                effects: vec![
                    Effect::CreateTokenCopy {
                        count: 1.into(),
                        of: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        exceptions: vec![],
                    },
                    Effect::CreateDelayedTrigger {
                        remember: Some(ObjectSelector::CreatedThisWay),
                        event: DelayedEvent::EndStep { your_turn: false },
                        frequency: DelayedFrequency::Once,
                        expiry: DelayedExpiry::Never,
                        effects: vec![Effect::Sacrifice {
                            what: ObjectSelector::Remembered,
                            optional: false,
                        }],
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
