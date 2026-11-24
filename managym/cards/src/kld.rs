//! Kaladesh.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
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
        cloudblazer(),
        brazen_scourge(),
        authority_of_the_consuls(),
        cultivators_caravan(),
        dramatic_reversal(),
        noxious_gearhulk(),
    ]
}

fn authority_of_the_consuls() -> CardDefinition {
    CardDefinition {
        key: "kld.authority_of_the_consuls",
        scryfall_id: "324b2f55-1e09-490e-8f7e-bfde85a91ac4",
        printed: PrintedCharacteristics {
            name: "Authority of the Consuls".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::OpponentsEnterTapped(CardPredicate::creature())],
            abilities: vec![Ability::when_enters(
                TriggerSubject::matching(Actor::Opponent, CardPredicate::creature(), false),
                vec![Effect::gain_life(1)],
            )],
            text_box: "Creatures your opponents control enter tapped.\nWhenever a creature an opponent controls enters, you gain 1 life.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
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

fn brazen_scourge() -> CardDefinition {
    let text_box =
        "Haste (This creature can attack and {T} as soon as it comes under your control.)";
    CardDefinition {
        key: "kld.brazen_scourge",
        scryfall_id: "68c6fbdb-7b5c-4ad0-88f5-4779deae16ce",
        printed: PrintedCharacteristics {
            name: "Brazen Scourge".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Gremlin".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Haste]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cloudblazer() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, you gain 2 life and draw two cards.";
    CardDefinition {
        key: "kld.cloudblazer",
        scryfall_id: "3cb12355-abd8-4bf3-aac1-f710ac162585",
        printed: PrintedCharacteristics {
            name: "Cloudblazer".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Scout".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::gain_life(2), Effect::DrawCards { count: 2.into() }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
