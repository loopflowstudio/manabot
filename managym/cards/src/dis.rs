//! Dissension.
use managym_state::ability::{
    Ability, Actor, Effect, ObjectSelector, PaymentCost, Per, TargetRequirement, TargetSpec,
    TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
};
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![trygon_predator(), macabre_waltz()]
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
                    to: Some(Actor::AnyPlayer),
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

fn macabre_waltz() -> CardDefinition {
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate::creature(),
    };
    let text_box = "Return up to two target creature cards from your graveyard to your hand, then discard a card.";
    CardDefinition {
        key: "dis.macabre_waltz",
        scryfall_id: "d9cd7bc3-73ba-4364-84b2-9954648cd8a9",
        printed: PrintedCharacteristics {
            name: "Macabre Waltz".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::return_to_hand(ObjectSelector::Targets(spec.clone())),
                Effect::discard(),
            ],
            targeting: vec![TargetRequirement::up_to(2, spec)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
