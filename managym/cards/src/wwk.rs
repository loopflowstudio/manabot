//! Worldwake.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, PaymentCost, PlayerSelector, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![kalastria_highborn()]
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
