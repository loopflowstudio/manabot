//! Ravnica Allegiance.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, Per, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![gateway_sneak()]
}

fn gateway_sneak() -> CardDefinition {
    let text_box = "Whenever a Gate you control enters, this creature can't be blocked this turn.\nWhenever this creature deals combat damage to a player, draw a card.";
    CardDefinition {
        key: "rna.gateway_sneak",
        scryfall_id: "edc0229d-05e6-41b7-b7a9-2a8b2b258add",
        printed: PrintedCharacteristics {
            name: "Gateway Sneak".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vedalken".into(), "Rogue".into()],
            power: Some(1),
            toughness: Some(3),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::you_control(CardPredicate::subtype("Gate")),
                    },
                    intervening_if: None,
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::CantBeBlocked,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::DealsCombatDamage {
                        subject: TriggerSubject::This,
                        to: Actor::AnyPlayer,
                        per: Per::Object,
                    },
                    intervening_if: None,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
