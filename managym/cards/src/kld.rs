//! Kaladesh.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, Scope,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        cultivators_caravan(),
        dramatic_reversal(),
        fumigate(),
        noxious_gearhulk(),
    ]
}

fn cultivators_caravan() -> CardDefinition {
    let text_box = "{T}: Add one mana of any color.\nCrew 3 (Tap any number of creatures you control with total power 3 or more: This Vehicle becomes an artifact creature until end of turn.)";
    CardDefinition {
        key: "kld.cultivators_caravan",
        scryfall_id: "b46b3726-4bc8-4e3a-bc6d-402c81663712",
        printed: PrintedCharacteristics {
            name: "Cultivator's Caravan".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Vehicle".into()],
            power: Some(5),
            toughness: Some(5),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    effects: vec![Effect::mana_choice(
                        ["W", "U", "B", "R", "G"].map(Mana::parse),
                    )],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    crew: Some(3),
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::BecomeCreature,
                    )],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dramatic_reversal() -> CardDefinition {
    let text_box = "Untap all nonland permanents you control.";
    CardDefinition {
        key: "kld.dramatic_reversal",
        scryfall_id: "dcb59045-2743-48ae-8063-727e551b1c41",
        printed: PrintedCharacteristics {
            name: "Dramatic Reversal".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::untap(ObjectSelector::each_you_control(
                CardPredicate {
                    not_card_types: vec![CardType::Land],
                    ..Default::default()
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fumigate() -> CardDefinition {
    let text_box = "Destroy all creatures. You gain 1 life for each creature destroyed this way.";
    CardDefinition {
        key: "kld.fumigate",
        scryfall_id: "f00f27a7-9e92-4fbf-baa8-f47a5eee48a6",
        printed: PrintedCharacteristics {
            name: "Fumigate".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: vec![
                Effect::destroy(ObjectSelector::each(
                    Scope::Battlefield,
                    CardPredicate::creature(),
                )),
                Effect::gain_life(EffectValue::Objects {
                    of: Box::new(ObjectSelector::Moved),
                    predicate: None,
                }),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn noxious_gearhulk() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature enters, you may destroy another target creature. If a creature is destroyed this way, you gain life equal to its toughness.";
    CardDefinition {
        key: "kld.noxious_gearhulk",
        scryfall_id: "9f86e5fe-8723-4494-b4cc-b7ac3a047bd1",
        printed: PrintedCharacteristics {
            name: "Noxious Gearhulk".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}{B}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Construct".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Menace]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        another: true,
                        ..TargetRequirement::one(TargetSpec::Creature)
                    },
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::default()),
                        if_paid: vec![
                            Effect::destroy(ObjectSelector::Target(TargetSpec::Creature)),
                            // Its last known toughness, only if it was destroyed.
                            Effect::If {
                                condition: Condition::Moved,
                                then: vec![Effect::gain_life(EffectValue::TargetToughness)],
                                otherwise: vec![],
                            },
                        ],
                        if_declined: vec![],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
