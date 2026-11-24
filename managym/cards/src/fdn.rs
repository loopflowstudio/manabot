//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, KeywordAction, ObjectSelector, PaymentCost,
        Per, PlayWindow, PlayerSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, ActivationZone, AdditionalCost, CardDefinition, CardType,
        CardTypes, Keyword, Keywords, PrintedCharacteristics, SacrificeCost, StaticAbility,
        TargetCostReduction,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        blasphemous_edict(),
        claws_out(),
        hungry_ghoul(),
        herald_of_eternal_dawn(),
        kellan_planar_trailblazer(),
        luminous_rebuke(),
        leyline_axe(),
        sire_of_seven_deaths(),
        high_fae_trickster(),
        sphinx_of_forgotten_lore(),
        suspicious_shambler(),
        tinybones_bauble_burglar(),
        loot_exuberant_explorer(),
        redcap_gutter_dweller(),
        zul_ashur_lich_lord(),
    ]
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
                                        to: Actor::AnyPlayer,
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
            activated_abilities: vec![ActivatedAbilityDefinition {
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{3}"),
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                }],
                ..Default::default()
            }],
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
