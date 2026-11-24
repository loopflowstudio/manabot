//! Ravnica Allegiance.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, Per, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![archway_angel(), gateway_sneak()]
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

fn archway_angel() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, you gain 2 life for each Gate you control.";
    CardDefinition {
        key: "rna.archway_angel",
        scryfall_id: "b209d219-b946-4226-a8b4-65a5f3837fac",
        printed: PrintedCharacteristics {
            name: "Archway Angel".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                // One gain of twice the Gates (CR 119.9), not one per Gate.
                effects: vec![Effect::gain_life(EffectValue::Scaled {
                    factor: 2,
                    value: Box::new(
                        ObjectSelector::each_you_control(CardPredicate::subtype("Gate")).count(),
                    ),
                })],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
