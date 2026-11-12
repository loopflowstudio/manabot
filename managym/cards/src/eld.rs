//! Throne of Eldraine.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, StaticAbility},
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        epic_downfall(),
        fabled_passage(),
        return_of_the_wildspeaker(),
        the_great_henge(),
    ]
}

fn epic_downfall() -> CardDefinition {
    let text_box = "Exile target creature with mana value 3 or greater.";

    CardDefinition {
        key: "eld.epic_downfall",
        scryfall_id: "63da83fe-fa59-40cb-a42e-e1b14b650bc8",
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
    }
}

fn fabled_passage() -> CardDefinition {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.";

    CardDefinition {
        key: "eld.fabled_passage",
        scryfall_id: "b841bfa8-7c17-4df2-8466-780ab9a4a53a",
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
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    token: Some(false),
                    ..creature()
                }),
            },
            effects: vec![
                Effect::put_counters(ObjectSelector::Triggering, CounterKind::PlusOnePlusOne, 1),
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
