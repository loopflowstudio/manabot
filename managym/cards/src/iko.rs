//! Ikoria: Lair of Behemoths.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, SacrificeCost, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        voracious_greatshark(),
        aegis_turtle(),
        bastion_of_remembrance(),
        drannith_magistrate(),
        weaponize_the_monsters(),
    ]
}

fn bastion_of_remembrance() -> CardDefinition {
    let text_box = "When this enchantment enters, create a 1/1 white Human Soldier creature token.\nWhenever a creature you control dies, each opponent loses 1 life and you gain 1 life.";

    CardDefinition {
        key: "iko.bastion_of_remembrance",
        scryfall_id: "2dd354dd-939e-4b1a-8ed6-fe89a7fd64bf",
        printed: PrintedCharacteristics {
            name: "Bastion of Remembrance".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::create_tokens("token.human_soldier", 1)],
                },
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::you_control(CardPredicate {
                            card_type: Some(CardType::Creature),
                            ..Default::default()
                        }),
                    },
                    effects: vec![Effect::DrainLife {
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

fn weaponize_the_monsters() -> CardDefinition {
    let text_box = "{2}, Sacrifice a creature: This enchantment deals 2 damage to any target.";

    CardDefinition {
        key: "iko.weaponize_the_monsters",
        scryfall_id: "68bba622-a0ab-4c0e-88b1-9120690ea5a0",
        printed: PrintedCharacteristics {
            name: "Weaponize the Monsters".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                    another: false,
                    or_mana: None,
                }),
                mana_cost: ManaCost::parse("{2}"),
                effects: vec![Effect::deal_damage(
                    2,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn drannith_magistrate() -> CardDefinition {
    let text_box = "Your opponents can't cast spells from anywhere other than their hands.";
    CardDefinition {
        key: "iko.drannith_magistrate",
        scryfall_id: "98b0a4a8-9319-451b-9b79-b0bca7a41e91",
        printed: PrintedCharacteristics {
            name: "Drannith Magistrate".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(3),
            statics: vec![StaticAbility::OpponentsCastOnlyFromHand],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn aegis_turtle() -> CardDefinition {
    CardDefinition {
        key: "iko.aegis_turtle",
        scryfall_id: "e433e7f0-7417-4dfe-a7a4-3f222b0a835f",
        printed: PrintedCharacteristics {
            name: "Aegis Turtle".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Turtle".into()],
            power: Some(0),
            toughness: Some(5),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn voracious_greatshark() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhen this creature enters, counter target artifact or creature spell.";
    CardDefinition {
        key: "iko.voracious_greatshark",
        scryfall_id: "1400155f-8911-45fd-aab2-998c8a28292c",
        printed: PrintedCharacteristics {
            name: "Voracious Greatshark".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Shark".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::CounterSpell {
                    what: ObjectSelector::Target(TargetSpec::SpellMatching {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Artifact, CardType::Creature],
                            ..Default::default()
                        },
                    }),
                    exile: false,
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
