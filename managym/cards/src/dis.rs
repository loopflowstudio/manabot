//! Dissension.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, PaymentCost, Per, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![trygon_predator()]
}

fn trygon_predator() -> CardDefinition {
    let text_box = "Flying\nWhenever this creature deals combat damage to a player, you may destroy target artifact or enchantment that player controls.";
    // Combat damage reaches only the defending player, so in a two-player
    // game "that player" is the opponent.
    let theirs = || TargetSpec::PermanentOpponentControls {
        predicate: CardPredicate {
            card_types_any: vec![CardType::Artifact, CardType::Enchantment],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "dis.trygon_predator",
        scryfall_id: "f31f54bf-7bf0-48f0-853d-1468713784eb",
        printed: PrintedCharacteristics {
            name: "Trygon Predator".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Beast".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Actor::AnyPlayer,
                    per: Per::Object,
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(theirs()),
                    // "You may": the catalog's zero-cost optional payment.
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::default()),
                        if_paid: vec![Effect::destroy(ObjectSelector::Target(theirs()))],
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
