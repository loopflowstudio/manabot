//! Worldwake.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, PaymentCost, PlayerSelector,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dragonmaster_outcast(), kalastria_highborn()]
}

fn kalastria_highborn() -> CardDefinition {
    let text_box = "Whenever this creature or another Vampire you control dies, you may pay {B}. If you do, target player loses 2 life and you gain 2 life.";
    let player = || PlayerSelector::Target(Box::new(TargetSpec::Player));
    CardDefinition {
        key: "wwk.kalastria_highborn",
        scryfall_id: "f1efd1dd-903c-47a0-b746-5571a3ea1755",
        printed: PrintedCharacteristics {
            name: "Kalastria Highborn".into(),
            mana_cost: Some(ManaCost::parse("{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Shaman".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate::subtype("Vampire")),
                },
                intervening_if: None,
                // The target is chosen as the ability is put on the stack
                // (CR 603.3d); the payment is offered as it resolves.
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Player),
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::parse("{B}")),
                        if_paid: vec![
                            Effect::LoseLife {
                                player: player(),
                                amount: EffectValue::Fixed(2),
                            },
                            Effect::gain_life(2),
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

fn dragonmaster_outcast() -> CardDefinition {
    let text_box = "At the beginning of your upkeep, if you control six or more lands, create a 5/5 red Dragon creature token with flying.";
    CardDefinition {
        key: "wwk.dragonmaster_outcast",
        scryfall_id: "c2297a4e-3c19-4748-9150-efbd2513066a",
        printed: PrintedCharacteristics {
            name: "Dragonmaster Outcast".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Shaman".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                intervening_if: Some(Condition::ControlsAtLeast {
                    count: 6,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    },
                    distinct_names: false,
                }),
                effects: vec![Effect::create_tokens("token.dragon_5_5", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
