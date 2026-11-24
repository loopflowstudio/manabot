//! Tarkir: Dragonstorm.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PaymentCost, PlayerSelector,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes,
        InsetSpell, Keyword, Keywords, PrintedCharacteristics, StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        riling_dawnbreaker(),
        marang_river_regent(),
        stormshriek_feral(),
        runescale_stormbrood(),
        whirlwing_stormbrood(),
        purging_stormbrood(),
        disruptive_stormbrood(),
        feral_deathgorger(),
        sagu_wildling(),
        dirgur_island_dragon(),
        teval_arbiter_of_virtue(),
        twinmaw_stormbrood(),
        scavenger_regent(),
    ]
}

fn scavenger_regent() -> CardDefinition {
    use managym_state::{
        ability::{ObjectSelector, PaymentCost, Scope},
        card::InsetSpell,
        effect::Modification,
    };
    let toxin = PrintedCharacteristics {
        name: "Exude Toxin".into(),
        mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::until_end_of_turn(
            ObjectSelector::each(Scope::Battlefield, CardPredicate {
                not_subtypes: vec!["Dragon".into()], ..CardPredicate::creature()
            }),
            Modification::ModifyPt { power: EffectValue::NegativeX, toughness: EffectValue::NegativeX },
        )],
        text_box: "Each non-Dragon creature gets -X/-X until end of turn. (Then shuffle this card into its owner's library.)".into(),
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.scavenger_regent",
        scryfall_id: "0d4b46a3-847a-44a7-9f68-2cb4657cad61",
        printed: PrintedCharacteristics {
            name: "Scavenger Regent".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::Ward(PaymentCost::Discard)],
            text_box: "Flying\nWard—Discard a card.".into(),
            inset: Some(InsetSpell::Omen(toxin.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn twinmaw_stormbrood() -> CardDefinition {
    use managym_state::ability::TriggerSubject;
    CardDefinition {
        key: "tdm.twinmaw_stormbrood",
        scryfall_id: "2999e3b1-6510-42b2-9429-28c07a64a44f",
        printed: PrintedCharacteristics {
            name: "Twinmaw Stormbrood".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::gain_life(5)],
            }],
            text_box: "Flying\nWhen this creature enters, you gain 5 life.".into(),
            inset: Some(InsetSpell::Omen(charring_bite().into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn charring_bite() -> PrintedCharacteristics {
    use managym_state::ability::{ObjectSelector, TargetSpec};
    let target = TargetSpec::Permanent {
        predicate: CardPredicate {
            flying: Some(false),
            ..CardPredicate::creature()
        },
    };
    PrintedCharacteristics {
        name: "Charring Bite".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::deal_damage(5, ObjectSelector::Target(target))],
        text_box: "Charring Bite deals 5 damage to target creature without flying. (Then shuffle this card into its owner's library.)".into(),
        ..Default::default()
    }
}

fn teval_arbiter_of_virtue() -> CardDefinition {
    let text_box = "Flying, lifelink\nSpells you cast have delve. (Each card you exile from your graveyard while casting those spells pays for {1}.)\nWhenever you cast a spell, you lose life equal to its mana value.";
    CardDefinition {
        key: "tdm.teval_arbiter_of_virtue",
        scryfall_id: "27a93f5b-7b32-49f0-a179-b897828fe49a",
        printed: PrintedCharacteristics {
            name: "Teval, Arbiter of Virtue".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Spirit".into(), "Dragon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
            statics: vec![StaticAbility::SpellsHave {
                predicate: CardPredicate::default(),
                cost: AdditionalCost::Delve,
            }],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::You,
                    amount: EffectValue::TriggeringManaValue,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dirgur_island_dragon() -> CardDefinition {
    let rules_text = "Tap up to one target creature. Draw a card. (Then shuffle this card into its owner's library.)";
    let other_face_text = "Flying\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";
    let inset = PrintedCharacteristics {
        name: "Skimming Strike".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                effects: vec![Effect::tap(ObjectSelector::Target(TargetSpec::Creature))],
            },
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
        ],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.dirgur_island_dragon",
        scryfall_id: "b1d21a9a-6b0c-4fbc-a427-81be885d326b",
        printed: PrintedCharacteristics {
            name: "Dirgur Island Dragon".into(),
            mana_cost: Some(ManaCost::parse("{5}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse(
                "{2}",
            )))],
            inset: Some(InsetSpell::Omen(inset.into())),
            text_box: other_face_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sagu_wildling() -> CardDefinition {
    let rules_text = "Search your library for a basic land card, reveal it, put it into your hand, then shuffle. (Also shuffle this card.)";
    let inset = PrintedCharacteristics {
        name: "Roost Seek".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::search_library(
            CardPredicate::basic_land(),
            EffectValue::Fixed(1),
            vec![Effect::return_to_hand(ObjectSelector::Selected)],
        )],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.sagu_wildling",
        scryfall_id: "d8b43b00-f4d1-436c-bf3f-6d414cd4ce38",
        printed: PrintedCharacteristics {
            name: "Sagu Wildling".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::gain_life(3)],
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            text_box: "Flying\nWhen this creature enters, you gain 3 life.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn disruptive_stormbrood() -> CardDefinition {
    let rules_text = "Destroy target creature with power 3 or less. (Then shuffle this card into its owner's library.)";
    let other_face_text =
        "Flying\nWhen this creature enters, destroy up to one target artifact or enchantment.";
    let target = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
            ..Default::default()
        },
    };
    let inset = PrintedCharacteristics {
        name: "Petty Revenge".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::destroy(ObjectSelector::Target(
            TargetSpec::Permanent {
                predicate: CardPredicate {
                    max_power: Some(3),
                    ..CardPredicate::creature()
                },
            },
        ))],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.disruptive_stormbrood",
        scryfall_id: "bd78e8ae-e927-40e7-9580-966c5e81f53c",
        printed: PrintedCharacteristics {
            name: "Disruptive Stormbrood".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::up_to(1, target.clone()),
                    effects: vec![Effect::destroy(ObjectSelector::Target(target))],
                }],
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            text_box: other_face_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn purging_stormbrood() -> CardDefinition {
    let rules_text = "Target creature gets +2/+2 and gains lifelink and hexproof until end of turn. (Then shuffle this card into its owner's library.)";
    let other_face_text = "Flying\nWard—Pay 2 life.\nWhen this creature enters, remove all counters from up to one target creature.";
    let inset = PrintedCharacteristics {
        name: "Absorb Essence".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::TargetClause {
            requirement: TargetRequirement::one(TargetSpec::Creature),
            effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(2, 2),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::AddKeywords(Keywords::of(&[
                        Keyword::Lifelink,
                        Keyword::Hexproof,
                    ])),
                ),
            ],
        }],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.purging_stormbrood",
        scryfall_id: "3988dc76-072c-4f43-849d-2e73c6f6ff58",
        printed: PrintedCharacteristics {
            name: "Purging Stormbrood".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::Ward(PaymentCost::Life(2))],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                    effects: vec![Effect::RemoveAllCounters {
                        from: ObjectSelector::Target(TargetSpec::Creature),
                    }],
                }],
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            text_box: other_face_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn whirlwing_stormbrood() -> CardDefinition {
    let rules_text = "Put three +1/+1 counters on target creature you control. (Then shuffle this card into its owner's library.)";
    let other_face_text =
        "Flash\nFlying\nYou may cast sorcery spells and Dragon spells as though they had flash.";
    let inset = PrintedCharacteristics {
        name: "Dynamic Soar".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::put_counters(
            ObjectSelector::Target(TargetSpec::CreatureYouControl),
            CounterKind::PlusOnePlusOne,
            3,
        )],
        text_box: rules_text.into(),
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.whirlwing_stormbrood",
        scryfall_id: "56a25eb1-bdb8-4f86-8d9a-3055ad1b2a13",
        printed: PrintedCharacteristics {
            name: "Whirlwing Stormbrood".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Flash]),
            statics: vec![StaticAbility::CastSpellsAsThoughFlash(CardPredicate {
                any_of: vec![
                    CardPredicate {
                        card_type: Some(CardType::Sorcery),
                        ..Default::default()
                    },
                    CardPredicate::subtype("Dragon"),
                ],
                ..Default::default()
            })],
            inset: Some(InsetSpell::Omen(inset.into())),
            text_box: other_face_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn riling_dawnbreaker() -> CardDefinition {
    let front_text="Flying, vigilance\nAt the beginning of combat on your turn, another target creature you control gets +1/+0 until end of turn.";
    let spell_text="Create a 2/2 white Soldier creature token. (Then shuffle this card into its owner's library.)";

    let inset = PrintedCharacteristics {
        name: "Signaling Roar".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        text_box: spell_text.into(),
        spell_effects: vec![Effect::create_tokens("token.white_soldier", 1)],
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.riling_dawnbreaker",
        scryfall_id: "312f7072-3bf8-449f-bfb7-93727ef26c66",
        printed: PrintedCharacteristics {
            name: "Riling Dawnbreaker".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance]),
            text_box: front_text.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        another: true,
                        ..TargetRequirement::one(TargetSpec::CreatureYouControl)
                    },
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        Modification::pump(1, 0),
                    )],
                }],
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn marang_river_regent() -> CardDefinition {
    let front_text="Flying\nWhen this creature enters, return up to two other target nonland permanents to their owners' hands.";
    let spell_text =
        "Draw three cards, then discard a card. (Then shuffle this card into its owner's library.)";
    let nonland = TargetSpec::Permanent {
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land],
            ..Default::default()
        },
    };
    let inset = PrintedCharacteristics {
        name: "Coil and Catch".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Omen".into()],
        text_box: spell_text.into(),
        spell_effects: vec![
            Effect::DrawCards {
                count: EffectValue::Fixed(3),
            },
            Effect::discard(),
        ],
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.marang_river_regent",
        scryfall_id: "f890bdc7-32e6-4492-bac7-7cabf54a8bfd",
        printed: PrintedCharacteristics {
            name: "Marang River Regent".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(6),
            toughness: Some(7),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: front_text.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        another: true,
                        ..TargetRequirement::up_to(2, nonland.clone())
                    },
                    effects: vec![Effect::return_to_hand(ObjectSelector::Targets(nonland))],
                }],
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn stormshriek_feral() -> CardDefinition {
    let front_text = "Flying, haste\n{1}{R}: This creature gets +1/+0 until end of turn.";
    let spell_text="Discard a card. If you do, draw two cards. (Then shuffle this card into its owner's library.)";

    let inset = PrintedCharacteristics {
        name: "Flush Out".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        text_box: spell_text.into(),
        spell_effects: vec![
            Effect::discard(),
            Effect::If {
                condition: managym_state::ability::Condition::CardsThisWayMatch(
                    CardPredicate::default(),
                ),
                then: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                }],
                otherwise: vec![],
            },
        ],
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.stormshriek_feral",
        scryfall_id: "0ec92c44-7cf0-48a5-a3ca-bc633496d887",
        printed: PrintedCharacteristics {
            name: "Stormshriek Feral".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Haste]),
            text_box: front_text.into(),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{R}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(1, 0),
                )],
                ..Default::default()
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn runescale_stormbrood() -> CardDefinition {
    let front_text="Flying\nWhenever you cast a noncreature spell or a Dragon spell, this creature gets +2/+0 until end of turn.";
    let spell_text="Counter target spell with mana value 2 or less. (Then shuffle this card into its owner's library.)";

    let inset = PrintedCharacteristics {
        name: "Chilling Screech".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Omen".into()],
        text_box: spell_text.into(),
        spell_effects: vec![Effect::CounterSpell {
            what: ObjectSelector::Target(TargetSpec::SpellMatching {
                predicate: CardPredicate {
                    max_mana_value: Some(2),
                    ..Default::default()
                },
            }),
            exile: false,
        }],
        ..Default::default()
    };
    CardDefinition {
        key: "tdm.runescale_stormbrood",
        scryfall_id: "317744d1-ed78-4b53-a4d8-8c7ecfd9c4ae",
        printed: PrintedCharacteristics {
            name: "Runescale Stormbrood".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(2),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: front_text.into(),
            abilities: vec![Ability::Triggered {
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
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(2, 0),
                )],
            }],
            inset: Some(InsetSpell::Omen(inset.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn feral_deathgorger() -> CardDefinition {
    let text = "Flying, deathtouch\nWhen this creature enters, exile up to two target cards from a single graveyard.";
    let dusk_sight = PrintedCharacteristics {
        name: "Dusk Sight".into(), mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Sorcery]), subtypes: vec!["Omen".into()],
        text_box: "Put a +1/+1 counter on up to one target creature. Draw a card. (Then shuffle this card into its owner's library.)".into(),
        spell_effects: vec![
            Effect::TargetClause {
                requirement: TargetRequirement::up_to(1,TargetSpec::Creature),
                effects: vec![Effect::put_counters(ObjectSelector::Target(TargetSpec::Creature), CounterKind::PlusOnePlusOne,1)],
            },
            Effect::DrawCards { count: 1.into() },
        ],
        ..Default::default()
    };
    let grave_card = TargetSpec::GraveyardCard {
        predicate: CardPredicate::default(),
    };
    CardDefinition {
        key: "tdm.feral_deathgorger",
        scryfall_id: "a147b94f-dfcf-44ce-8a73-b2fe6c4efc0e",
        printed: PrintedCharacteristics {
            name: "Feral Deathgorger".into(),
            mana_cost: Some(ManaCost::parse("{5}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Deathtouch]),
            text_box: text.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        same_graveyard: true,
                        ..TargetRequirement::up_to(2, grave_card.clone())
                    },
                    effects: vec![Effect::exile(ObjectSelector::Targets(grave_card))],
                }],
            }],
            inset: Some(InsetSpell::Omen(dusk_sight.into())),
            ..Default::default()
        },
        ..Default::default()
    }
}
