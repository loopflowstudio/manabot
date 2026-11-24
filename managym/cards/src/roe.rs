use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        wildheart_invoker(),
        pelakka_wurm(),
        fleeting_distraction(),
        battle_rattle_shaman(),
        evolving_wilds(),
        prophetic_prism(),
        training_grounds(),
    ]
}

fn prophetic_prism() -> CardDefinition {
    let text_box = "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color.";

    CardDefinition {
        key: "roe.prophetic_prism",
        scryfall_id: "cfb90d44-8cb1-4b83-b2f2-92c19d6304fb",
        printed: PrintedCharacteristics {
            name: "Prophetic Prism".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effects: vec![Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(Mana::parse),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn training_grounds() -> CardDefinition {
    let text_box = "Activated abilities of creatures you control cost {2} less to activate. This effect can't reduce the mana in that cost to less than one mana.";
    CardDefinition {
        key: "roe.training_grounds",
        scryfall_id: "e2cf16f8-6e69-46b3-8453-1d1a2a5670e2",
        printed: PrintedCharacteristics {
            name: "Training Grounds".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::ActivatedAbilitiesCostLess {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                generic: 2,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn evolving_wilds() -> CardDefinition {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.";
    CardDefinition {
        key: "roe.evolving_wilds",
        scryfall_id: "bc7e0407-fea1-43ef-8580-82271e440bb3",
        printed: PrintedCharacteristics {
            name: "Evolving Wilds".into(),
            types: CardTypes::new([CardType::Land]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::search_library(
                    CardPredicate::basic_land(),
                    EffectValue::Fixed(1),
                    vec![Effect::return_to_battlefield(
                        ObjectSelector::Selected,
                        true,
                    )],
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn battle_rattle_shaman() -> CardDefinition {
    let text_box = "At the beginning of combat on your turn, you may have target creature get +2/+0 until end of turn.";
    CardDefinition {
        key: "roe.battle_rattle_shaman",
        scryfall_id: "aa1df08a-ccef-44cf-936a-838e238c27c1",
        printed: PrintedCharacteristics {
            name: "Battle-Rattle Shaman".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Shaman".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Creature),
                    effects: vec![Effect::Modal {
                        modes: vec![
                            vec![Effect::until_end_of_turn(
                                ObjectSelector::Target(TargetSpec::Creature),
                                Modification::pump(2, 0),
                            )],
                            vec![],
                        ],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fleeting_distraction() -> CardDefinition {
    let text_box = "Target creature gets -1/-0 until end of turn.\nDraw a card.";
    CardDefinition {
        key: "roe.fleeting_distraction",
        scryfall_id: "ed843c4d-28b5-4a4c-8bae-8f03f329bf2b",
        printed: PrintedCharacteristics {
            name: "Fleeting Distraction".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(-1, 0),
                ),
                Effect::DrawCards { count: 1.into() },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pelakka_wurm() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nWhen this creature enters, you gain 7 life.\nWhen this creature dies, draw a card.";
    CardDefinition {
        key: "roe.pelakka_wurm",
        scryfall_id: "8e732593-0bdc-4dd4-9b07-9aa1a780e6e8",
        printed: PrintedCharacteristics {
            name: "Pelakka Wurm".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Wurm".into()],
            power: Some(7),
            toughness: Some(7),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![
                Ability::when_enters(TriggerSubject::This, vec![Effect::gain_life(7)]),
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::DrawCards { count: 1.into() }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wildheart_invoker() -> CardDefinition {
    let text_box = "{8}: Target creature gets +5/+5 and gains trample until end of turn. (It can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "roe.wildheart_invoker",
        scryfall_id: "dc8315bf-03af-4f19-92c7-556e486cb099",
        printed: PrintedCharacteristics {
            name: "Wildheart Invoker".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Shaman".into()],
            power: Some(4),
            toughness: Some(3),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{8}"),
                effects: Effect::pump_and_grant_until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    5,
                    5,
                    Keyword::Trample,
                ),
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
