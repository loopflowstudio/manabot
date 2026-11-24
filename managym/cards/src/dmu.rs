//! Dominaria United.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, PaymentCost, PlayerSelector, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bite_down(), garna_bloodfist_of_keld(), tolarian_terror()]
}

fn garna_bloodfist_of_keld() -> CardDefinition {
    let text_box = "Whenever another creature you control dies, draw a card if it was attacking. Otherwise, Garna deals 1 damage to each opponent.";
    CardDefinition {
        key: "dmu.garna_bloodfist_of_keld",
        scryfall_id: "294c5f08-08e7-458f-8838-ff321dc5d9f2",
        printed: PrintedCharacteristics {
            name: "Garna, Bloodfist of Keld".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Berserker".into()],
            power: Some(4),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::another_you_control(CardPredicate::creature()),
                },
                intervening_if: None,
                // Whether it was attacking is its last known information
                // (CR 603.10a), read as the ability resolves.
                effects: vec![Effect::If {
                    condition: Condition::TriggeringMatches(CardPredicate {
                        attacking: true,
                        ..CardPredicate::default()
                    }),
                    then: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                    otherwise: vec![Effect::DealDamageToPlayer {
                        player: PlayerSelector::EachOpponent,
                        amount: EffectValue::Fixed(1),
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tolarian_terror() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast for each instant and sorcery card in your graveyard.\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";
    CardDefinition {
        key: "dmu.tolarian_terror",
        scryfall_id: "42f01cba-43d4-46ad-b7a5-d7631b0e1347",
        printed: PrintedCharacteristics {
            name: "Tolarian Terror".into(),
            mana_cost: Some(ManaCost::parse("{6}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Serpent".into()],
            power: Some(5),
            toughness: Some(5),
            statics: vec![
                StaticAbility::CostsLessPerGraveyardCard(CardPredicate {
                    card_types_any: vec![CardType::Instant, CardType::Sorcery],
                    ..Default::default()
                }),
                StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse("{2}"))),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bite_down() -> CardDefinition {
    let text_box = "Target creature you control deals damage equal to its power to target creature or planeswalker you don't control.";
    CardDefinition {
        key: "dmu.bite_down",
        scryfall_id: "0eacd3de-b803-4322-8d88-d533761aa748",
        printed: PrintedCharacteristics {
            name: "Bite Down".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![
                TargetRequirement::one(TargetSpec::CreatureYouControl),
                TargetRequirement::one(TargetSpec::PermanentOpponentControls {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                        ..Default::default()
                    },
                }),
            ],
            spell_effects: vec![Effect::TargetCreaturesDealPowerDamageToLastTarget],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
