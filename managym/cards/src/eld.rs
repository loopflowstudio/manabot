//! Throne of Eldraine.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics, StaticAbility,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        raging_redcap(),
        epic_downfall(),
        fabled_passage(),
        return_of_the_wildspeaker(),
        the_great_henge(),
        thrill_of_possibility(),
        wildborn_preserver(),
    ]
}

fn epic_downfall() -> CardDefinition {
    let text_box = "Exile target creature with mana value 3 or greater.";

    CardDefinition {
        key: "eld.epic_downfall",
        scryfall_id: "63da83fe-fa59-40cb-a42e-e1b14b650bc8",
        printed: PrintedCharacteristics {
            name: "Epic Downfall".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::exile(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        min_mana_value: Some(3),
                        ..Default::default()
                    },
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fabled_passage() -> CardDefinition {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.";

    CardDefinition {
        key: "eld.fabled_passage",
        scryfall_id: "b841bfa8-7c17-4df2-8466-780ab9a4a53a",
        printed: PrintedCharacteristics {
            name: "Fabled Passage".into(),
            types: CardTypes::new([CardType::Land]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::SelectCards {
                    from: ObjectSelector::Zone(ZoneType::Library),
                    predicate: CardPredicate::basic_land(),
                    min: 0,
                    max: EffectValue::Fixed(1),
                    effects: vec![
                        Effect::return_to_battlefield(ObjectSelector::Selected, true),
                        Effect::ShuffleLibrary,
                        Effect::If {
                            condition: Condition::ControlsAtLeast {
                                count: 4,
                                predicate: CardPredicate {
                                    card_type: Some(CardType::Land),
                                    ..Default::default()
                                },
                            },
                            then: vec![Effect::Untap {
                                what: ObjectSelector::Moved,
                            }],
                            otherwise: vec![],
                        },
                    ],
                    if_declined: vec![Effect::ShuffleLibrary],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn return_of_the_wildspeaker() -> CardDefinition {
    let text_box = "Choose one —\n• Draw cards equal to the greatest power among non-Human creatures you control.\n• Non-Human creatures you control get +3/+3 until end of turn.";
    let non_human = || CardPredicate {
        card_type: Some(CardType::Creature),
        not_subtypes: vec!["Human".into()],
        ..Default::default()
    };
    CardDefinition {
        key: "eld.return_of_the_wildspeaker",
        scryfall_id: "b88a4943-bd1b-4d10-9cd3-b2ab91b25c10",
        printed: PrintedCharacteristics {
            name: "Return of the Wildspeaker".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::DrawCards {
                        count: EffectValue::GreatestPowerYouControl {
                            predicate: non_human(),
                        },
                    }],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each_you_control(non_human()),
                        Modification::pump(3, 3),
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn the_great_henge() -> CardDefinition {
    let text_box = "This spell costs {X} less to cast, where X is the greatest power among creatures you control.\n{T}: Add {G}{G}. You gain 2 life.\nWhenever a nontoken creature you control enters, put a +1/+1 counter on it and draw a card.";
    let creature = || CardPredicate {
        card_type: Some(CardType::Creature),
        ..Default::default()
    };
    CardDefinition {
        key: "eld.the_great_henge",
        scryfall_id: "af915ed2-1f34-43f6-85f5-2430325b720f",
        printed: PrintedCharacteristics {
            name: "The Great Henge".into(),
            mana_cost: Some(ManaCost::parse("{7}{G}{G}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            statics: vec![StaticAbility::CostsLessByGreatestPower(creature())],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::parse("GG")), Effect::gain_life(2)],
                ..Default::default()
            }],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        token: Some(false),
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![
                    Effect::put_counters(
                        ObjectSelector::Triggering,
                        CounterKind::PlusOnePlusOne,
                        1,
                    ),
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wildborn_preserver() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nReach (This creature can block creatures with flying.)\nWhenever another non-Human creature you control enters, you may pay {X}. When you do, put X +1/+1 counters on this creature.";
    CardDefinition {
        key: "eld.wildborn_preserver",
        scryfall_id: "55f76830-369e-4224-9ded-7d1ce04c87e4",
        printed: PrintedCharacteristics {
            name: "Wildborn Preserver".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Archer".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flash, Keyword::Reach]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate {
                        not_subtypes: vec!["Human".into()],
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![Effect::OptionalManaPaymentX {
                    if_paid: vec![Effect::WhenYouDo {
                        effects: vec![Effect::put_counters(
                            ObjectSelector::This,
                            CounterKind::PlusOnePlusOne,
                            EffectValue::X,
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

fn thrill_of_possibility() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, discard a card.\nDraw two cards.";
    CardDefinition {
        key: "eld.thrill_of_possibility",
        scryfall_id: "c9021f85-7ab4-4a78-a398-1611fe09cd14",
        printed: PrintedCharacteristics {
            name: "Thrill of Possibility".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Discard],
            spell_effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(2),
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn raging_redcap() -> CardDefinition {
    let text_box =
        "Double strike (This creature deals both first-strike and regular combat damage.)";
    CardDefinition {
        key: "eld.raging_redcap",
        scryfall_id: "d9325398-41c3-4177-a64d-ea38cb7a8737",
        printed: PrintedCharacteristics {
            name: "Raging Redcap".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Knight".into()],
            power: Some(1),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::DoubleStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
